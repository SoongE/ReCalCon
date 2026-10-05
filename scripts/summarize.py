from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

OOD_DATASETS = ["imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_sketch", "objectnet"]

SHORT = {
    "imagenet": "ID", "imagenet_r": "R", "imagenet_a": "A", "imagenet_v2": "V2",
    "imagenet_sketch": "Ske.", "objectnet": "Obj.",
}


def harmonic_mean(a, b):
    return 2 * a * b / (a + b) if (a + b) else 0.0


def collect(root):
    rows = list()
    for path in sorted(Path(root).glob("*/results.csv")):
        df = pd.read_csv(path)
        row = {"run": path.parent.name}
        row.update(dict(zip(df["dataset"], df["score"])))

        ood = [row[d] for d in OOD_DATASETS if d in row]
        if ood and "imagenet" in row:
            row["O.M."] = sum(ood) / len(ood)
            row["H.M."] = harmonic_mean(row["imagenet"], row["O.M."])
        elif len(df) > 1:
            row["Mean"] = df["score"].mean()

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize ReCalCon results")
    parser.add_argument("root", nargs="?", default="outputs", help="directory holding <run>/results.csv")
    parser.add_argument("--sort", default=None, help="column to sort by (default: last column)")
    args = parser.parse_args()

    df = collect(args.root)
    if df.empty:
        raise SystemExit(f"no results found under {args.root}/*/results.csv")

    sort_key = args.sort or df.columns[-1]
    df = df.sort_values(by=sort_key, ascending=False)

    table = Table(show_header=True, header_style="bold magenta")
    for column in df.columns:
        table.add_column(SHORT.get(column, column), justify="left" if column == "run" else "center")
    for _, row in df.iterrows():
        table.add_row(*[f"{v:.1f}" if isinstance(v, float) else str(v) for v in row])

    Console().print(table)
    df.to_csv(Path(args.root) / "summary.csv", index=False)


if __name__ == "__main__":
    main()
