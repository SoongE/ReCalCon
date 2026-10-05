from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from recalcon.config import Config, parse_cli_overrides
from recalcon.data import create_dataloader, create_dataset, resolve_group
from recalcon.engine import Evaluator, FewShotEvaluator
from recalcon.losses import build_criterion
from recalcon.models import build_model
from recalcon.registry import get_trainer_cls
from recalcon.utils import build_optimizer_and_scheduler, check_batch_size, resume, setup_accelerator


def evaluate(cfg, accelerator, model, tokenizer):
    rows = list()
    for name in resolve_group(cfg.data.eval):
        dataset = create_dataset(cfg, name, is_train=False)

        if cfg.data.n_shot > 0:
            train_dataset = create_dataset(cfg, name, split="train", is_train=True,
                                           paired_views=True, n_shot=cfg.data.n_shot)
            val_dataset = create_dataset(cfg, name, split="val", is_train=False)
            evaluator = FewShotEvaluator(cfg, accelerator, model, tokenizer, dataset, name,
                                         train_dataset=train_dataset, val_dataset=val_dataset)
        else:
            evaluator = Evaluator(cfg, accelerator, model, tokenizer, dataset)

        metrics = evaluator()
        rows.append(dict(dataset=name, shot=cfg.data.n_shot, **metrics))
        accelerator.print(rows[-1])

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train ReCalCon")
    parser.add_argument("--config", required=True, help="short name or path to a YAML config")
    args, rest = parser.parse_known_args()

    cfg = Config.from_name(args.config).apply_overrides(parse_cli_overrides(rest))

    accelerator = setup_accelerator(cfg)
    check_batch_size(cfg, accelerator.num_processes)

    output_dir = Path(cfg.train.output_dir)
    if accelerator.is_main_process:
        cfg.to_yaml(output_dir / "config.yaml")

    model, tokenizer = build_model(cfg)
    model.to(accelerator.device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)

    train_dataset = create_dataset(cfg, cfg.data.train, split="train", is_train=True,
                                   paired_views=True, n_shot=cfg.data.n_shot)
    train_loader = create_dataloader(train_dataset, cfg.train.batch_size, cfg.data.num_workers, is_train=True)

    val_loader = None
    if cfg.data.n_shot > 0:
        val_dataset = create_dataset(cfg, cfg.data.train, split="val", is_train=False)
        val_loader = create_dataloader(val_dataset, cfg.eval.batch_size, cfg.eval.num_workers, is_train=False)

    optimizer, scheduler, num_epochs = build_optimizer_and_scheduler(cfg, model, len(train_loader))
    criterion = build_criterion(cfg)

    start_epoch = resume(model, optimizer, scheduler, cfg, accelerator)
    model, optimizer = accelerator.prepare(model, optimizer)
    train_loader = accelerator.prepare_data_loader(train_loader, device_placement=False)
    if val_loader is not None:
        val_loader = accelerator.prepare_data_loader(val_loader, device_placement=False)

    trainer = get_trainer_cls(cfg.model.name)(cfg, accelerator, model, tokenizer, (train_loader, val_loader),
                                              criterion, optimizer, scheduler, (start_epoch, num_epochs))
    trainer()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.data.n_shot > 0:
            best = torch.load(str(output_dir / "best.ckpt"), map_location="cpu")["state_dict"]
            accelerator.unwrap_model(model).load_state_dict(best)

        df = evaluate(cfg, accelerator, model, tokenizer)
        df.to_csv(output_dir / "results.csv", index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
