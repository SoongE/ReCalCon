from __future__ import annotations

import shutil
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
from accelerate.utils import send_to_device
from torch.nn.functional import normalize

from ..registry import register_trainer
from ..utils import ConsoleCallback, RunLogger, RunningMean, sync_buffer


@register_trainer("recalcon")
class Trainer:
    def __init__(self, cfg, accelerator, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.world_size = accelerator.num_processes
        self.distributed = accelerator.num_processes > 1

        self.model = model
        self.module = accelerator.unwrap_model(model)
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader, self.val_loader = loaders
        self.start_epoch, self.num_epochs = epochs

        self.output_dir = Path(cfg.train.output_dir)
        self.grad_accumulation = cfg.train.grad_accumulation
        self.log_interval = cfg.train.log_interval
        self.dist_bn = cfg.train.dist_bn
        self.sample_size = cfg.train.batch_size * cfg.train.grad_accumulation * self.world_size

        self.criteria_metric = cfg.train.criteria_metric
        self.decreasing = cfg.train.criteria_decreasing
        self.best_metric = float("inf") if self.decreasing else 0.0
        self.best_epoch = 0

        self.logger = RunLogger(cfg, self.output_dir)
        self.losses = RunningMean(self.device)
        self.duration = RunningMean(self.device)
        self.data_duration = RunningMean(self.device)

        if hasattr(criterion, "rank"):
            criterion.rank = accelerator.local_process_index
            criterion.world_size = accelerator.num_processes

        dataset = self.train_loader.dataset
        if hasattr(dataset, "setup_prompt_transform"):
            dataset.setup_prompt_transform()

        self.fusion_alpha = cfg.loss.fusion_alpha
        self.image_fusion_threshold = 1.0 - cfg.loss.image_fusion_prob
        if self.fusion_alpha > 0:
            self._beta = torch.distributions.Beta(torch.tensor([self.fusion_alpha]),
                                                  torch.tensor([self.fusion_alpha]))

    def sample_lam(self):
        lam = torch.tensor(float(self._beta.sample()), device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(lam, 0)
        return lam

    def sample_uniform(self):
        value = torch.rand(1, device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(value, 0)
        return value.item()

    def draw_lams(self):
        if self.fusion_alpha <= 0:
            return 1.0, 1.0
        if self.sample_uniform() > self.image_fusion_threshold:
            return self.sample_lam(), 1.0
        return 1.0, self.sample_lam()

    def iterate(self, data):
        x, aug_x, y, prompt, aug_prompt = data

        prompt = torch.concat([self.tokenizer(prompt), self.tokenizer(aug_prompt)])
        prompt = prompt.to(self.device, non_blocking=True)
        x = torch.concat([x, aug_x]).to(self.device, non_blocking=True, memory_format=torch.channels_last)
        y = torch.concat([y, y]).to(self.device, non_blocking=True)

        img_lam, txt_lam = self.draw_lams()

        with self.accelerator.autocast():
            image_features, text_features = self.model(x, prompt, img_lam=img_lam, txt_lam=txt_lam)
            loss = self.criterion(image_features, text_features, y,
                                  self.module.logit_scale.exp(), min(img_lam, txt_lam))
        return loss

    def train(self, epoch):
        self.losses.reset()
        self.duration.reset()
        self.data_duration.reset()

        total_len = len(self.train_loader)
        accum_steps = self.grad_accumulation
        update_len = (total_len + accum_steps - 1) // accum_steps
        num_updates = epoch * update_len
        last = total_len - 1

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_start = start = perf_counter()

        for i, data in enumerate(self.train_loader):
            self.data_duration.update(perf_counter() - data_start)
            is_accumulating = (i + 1) % accum_steps != 0 and i != last
            update_idx = i // accum_steps

            sync = self.accelerator.no_sync(self.model) if is_accumulating and self.distributed else nullcontext()
            with sync:
                loss = self.iterate(data)
                self.accelerator.backward(loss)

            self.losses.update(loss)

            if is_accumulating:
                data_start = perf_counter()
                continue

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.duration.update(perf_counter() - start)
            start = perf_counter()
            num_updates += 1

            if update_idx % self.log_interval == 0 or i == last:
                lrl = [group["lr"] for group in self.optimizer.param_groups]
                ConsoleCallback.on_train(epoch, update_idx, update_len, loss, sum(lrl) / len(lrl),
                                         self.duration.compute(), self.sample_size,
                                         self.data_duration.compute())

            self.scheduler.step_update(num_updates=num_updates, metric=self.losses.compute())
            data_start = perf_counter()

        return {"loss": self.reduce(self.losses.compute())}

    def reduce(self, value):
        if not self.distributed:
            return value
        tensor = torch.tensor(value, device=self.device)
        return self.accelerator.reduce(tensor, reduction="mean").item()

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        dataset = self.val_loader.dataset

        text_classifier = list()
        for class_name in dataset.class_name:
            if hasattr(dataset, "get_gpt_text"):
                text = dataset.get_gpt_text(class_name)
            else:
                text = [t.format(name=class_name) for t in dataset.prompt]
            tokens = self.tokenizer(text).to(self.device)
            with self.accelerator.autocast():
                feature = self.module.encode_text(tokens)
            feature = normalize(feature, dim=-1).mean(0)
            text_classifier.append(feature / feature.norm())
        text_classifier = torch.stack(text_classifier, dim=0).to(self.device)

        features, labels = list(), list()
        for data in self.val_loader:
            x, y = data[0], data[1]
            x = x.to(self.device, memory_format=torch.channels_last)
            with self.accelerator.autocast():
                features.append(self.module.encode_image(x).detach())
            labels.append(y.to(self.device))

        features = normalize(torch.cat(features, dim=0), dim=-1)
        labels = torch.cat(labels)
        logits = 100.0 * features @ text_classifier.mT
        return {"Top1": logits.argmax(dim=-1).eq(labels).float().mean().item()}

    def _distribute_bn(self):
        if not self.distributed or self.dist_bn not in ("broadcast", "reduce"):
            return
        for name, buffer in self.module.named_buffers(recurse=True):
            if "running_mean" in name or "running_var" in name:
                sync_buffer(buffer, self.dist_bn)

    def _save(self, epoch, metric):
        latest = self.output_dir / "latest.ckpt"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save(send_to_device({
                "epoch": epoch,
                "model": self.cfg.model.name,
                "state_dict": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.to_dict(),
                self.criteria_metric: metric,
            }, "cpu"), latest)

        improved = metric < self.best_metric if self.decreasing else metric > self.best_metric
        if improved:
            self.best_metric, self.best_epoch = metric, epoch
            if self.accelerator.is_main_process:
                shutil.copy(latest, self.output_dir / "best.ckpt")

    def __call__(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.distributed:
                self.set_epoch(epoch)

            train_metrics = self.train(epoch)
            self._distribute_bn()

            eval_metrics = dict()
            if self.val_loader is not None:
                eval_metrics = self.validate(epoch)
                ConsoleCallback.on_eval(eval_metrics, epoch)

            self.scheduler.step(epoch + 1)

            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"eval_{k}": v for k, v in eval_metrics.items()}}
            self._save(epoch, metrics.get(f"eval_{self.criteria_metric}",
                                          metrics.get(f"train_{self.criteria_metric}")))
            metrics[f"best_{self.criteria_metric}"] = self.best_metric
            self.logger.log_metrics(metrics, epoch)
            ConsoleCallback.on_epoch(self.criteria_metric, self.best_metric, self.best_epoch)

    def set_epoch(self, epoch):
        loader = self.train_loader
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        elif hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
