from __future__ import annotations

import csv
import functools
import logging
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from safetensors.torch import load_file
from termcolor import colored
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

MIXED_PRECISION = {
    "16-mixed": "fp16",
    "16": "fp16",
    "bf16-mixed": "bf16",
    "bf16": "bf16",
    "32-true": "no",
    "32": "no",
}

LAUNCHED = ("LOCAL_RANK", "ACCELERATE_TORCH_DEVICE", "TORCHELASTIC_RUN_ID")


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def rank_zero_only(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) if is_main_process() else None

    return wrapper


def seed_everything(seed, local_rank=0):
    each_seed = seed + local_rank
    random.seed(each_seed)
    np.random.seed(each_seed)
    torch.manual_seed(each_seed)
    torch.cuda.manual_seed_all(each_seed)


def setup_accelerator(cfg):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    gpus = cfg.train.gpus if isinstance(cfg.train.gpus, list) else [cfg.train.gpus]
    if not any(key in os.environ for key in LAUNCHED):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision=MIXED_PRECISION[cfg.train.precision])
    seed_everything(cfg.train.seed, accelerator.local_process_index)

    setup_logging(output_dir, accelerator.is_main_process)
    return accelerator


def setup_logging(output_dir, is_master):
    handlers = [logging.StreamHandler()] if is_master else []
    if is_master:
        handlers.append(logging.FileHandler(Path(output_dir) / "train.log"))
    logging.basicConfig(level=logging.INFO if is_master else logging.ERROR,
                        format="[%(levelname)s] - %(message)s", handlers=handlers, force=True)


class RunLogger:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.use_wandb = cfg.train.wandb
        self.csv_path = Path(output_dir) / "summary.csv"
        self.write_header = not self.csv_path.exists()

        if self.use_wandb:
            self._init_wandb()

    @rank_zero_only
    def _init_wandb(self):
        import wandb

        wandb.init(project="recalcon", config=self.cfg.to_dict(),
                   name=Path(self.cfg.train.output_dir).name,
                   settings=wandb.Settings(_disable_stats=True), resume="allow")

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if self.use_wandb:
            import wandb

            wandb.log(metrics)
        metrics = dict(metrics, epoch=step)
        with open(self.csv_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            if self.write_header:
                writer.writeheader()
            writer.writerow(metrics)
        self.write_header = False


class ConsoleCallback:
    @staticmethod
    @rank_zero_only
    def on_epoch(metric_name, best_metric, best_epoch):
        logging.info(colored(f"*** Best {metric_name}: {best_metric:.5f} @ epoch {best_epoch} ***", "blue"))

    @staticmethod
    @rank_zero_only
    def on_train(epoch, update_idx, updates_per_epoch, loss, lr, duration, batch_size, data_duration):
        eta = timedelta(seconds=int((updates_per_epoch - update_idx) * duration))
        logging.info(f'{"Train":>5}: {epoch:>3} [{update_idx:>4d}/{updates_per_epoch - 1}] '
                     f"({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  "
                     f"Loss: {loss.item():#.3g}  "
                     f"LR: {lr:.3e}  "
                     f"Data: {data_duration:>5.2f}s  "
                     f"Batch: {duration:>5.2f}s  "
                     f"TP: {batch_size / duration:>7.2f}/s  "
                     + colored(f"ETA: {eta}  ", "yellow"))

    @staticmethod
    @rank_zero_only
    def on_eval(metrics, epoch):
        log = f'{"Eval":>5}: {epoch:>3}  '
        for k, v in metrics.items():
            log += f"{k}: {float(v):.4f}  "
        logging.info(colored(log.rstrip(), "green"))


def param_groups(model, adapter_lr=None, prompt_lr=None):
    adapter, backbone, prompt = list(), list(), list()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "adapter" in name:
            adapter.append(param)
        elif "learnable_context" in name:
            prompt.append(param)
        else:
            backbone.append(param)

    groups = [{"params": adapter}, {"params": backbone}, {"params": prompt}]
    if adapter_lr is not None:
        groups[0]["lr"] = adapter_lr
    if prompt_lr is not None:
        groups[2]["lr"] = prompt_lr
    return groups


def build_optimizer_and_scheduler(cfg, model, iter_per_epoch):
    t = cfg.train
    optimizer = create_optimizer_v2(
        param_groups(model, t.adapter_lr, t.prompt_lr),
        opt=t.opt,
        lr=t.lr,
        weight_decay=t.weight_decay,
        eps=t.opt_eps,
        betas=tuple(t.opt_betas),
    )

    updates_per_epoch = (iter_per_epoch + t.grad_accumulation - 1) // t.grad_accumulation
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=t.sched,
        num_epochs=t.epochs,
        warmup_epochs=t.warmup_epochs,
        cooldown_epochs=t.cooldown_epochs,
        warmup_lr=t.warmup_lr,
        min_lr=t.min_lr,
        step_on_epochs=not t.sched_on_updates,
        updates_per_epoch=updates_per_epoch,
    )
    return optimizer, scheduler, num_epochs


def resume(model, optimizer, scheduler, cfg, accelerator):
    path = cfg.train.resume
    if not path:
        return 0

    path = Path(path)
    if path.is_dir():
        path = path / "latest.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    scheduler.step(start_epoch)
    accelerator.print(f"resumed from {path} at epoch {start_epoch}")
    return start_epoch


def load_weights(model, checkpoint):
    if str(checkpoint).endswith(".safetensors"):
        state = load_file(str(checkpoint))
    else:
        state = torch.load(str(checkpoint), map_location="cpu")
        state = state.get("state_dict", state)
    model.load_state_dict(state)
    return model


def sync_buffer(buffer, mode):
    if mode == "reduce":
        dist.all_reduce(buffer, op=dist.ReduceOp.AVG)
    else:
        dist.broadcast(buffer, 0)


class RunningMean:
    def __init__(self, device=None):
        self.device = device
        self.reset()

    def reset(self):
        self.total = torch.zeros((), device=self.device)
        self.count = 0

    def update(self, value):
        self.total += value.detach() if torch.is_tensor(value) else value
        self.count += 1

    def compute(self):
        if self.count == 0:
            return 0.0
        return (self.total / self.count).item()


class TopOneAccuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, labels):
        self.correct += int(logits.argmax(dim=-1).eq(labels).sum())
        self.total += int(labels.numel())

    def compute(self):
        return self.correct / max(self.total, 1)


def check_batch_size(cfg, world_size):
    effective = cfg.train.batch_size * world_size * cfg.train.grad_accumulation
    if effective != cfg.train.total_batch:
        print(colored("[WARNING]", "red"),
              f"effective batch {effective} (batch_size {cfg.train.batch_size} x {world_size} gpus "
              f"x {cfg.train.grad_accumulation} accum) differs from train.total_batch "
              f"{cfg.train.total_batch}; contrastive results depend on it")
