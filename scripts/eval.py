from __future__ import annotations

import argparse
from pathlib import Path

import torch

from recalcon.config import Config, parse_cli_overrides
from recalcon.models import build_model
from recalcon.utils import load_weights, setup_accelerator
from scripts.train import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ReCalCon checkpoint")
    parser.add_argument("--config", required=True, help="short name or path to a YAML config")
    args, rest = parser.parse_known_args()

    cfg = Config.from_name(args.config).apply_overrides(parse_cli_overrides(rest))
    accelerator = setup_accelerator(cfg)

    model, tokenizer = build_model(cfg)
    if cfg.eval.checkpoint:
        load_weights(model, cfg.eval.checkpoint)
    model.to(accelerator.device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)

    df = evaluate(cfg, accelerator, model, tokenizer)
    output = Path(cfg.train.output_dir) / "results.csv"
    df.to_csv(output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
