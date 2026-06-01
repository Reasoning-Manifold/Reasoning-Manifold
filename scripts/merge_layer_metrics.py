"""Aggregate per-layer metrics CSVs from a directory tree.

Replaces the family-specific clones:
    result_qwen3_stride1/merge_data.py
    result_qwen3_stride1/merge_gemma_data.py
    result_qwen2_instruct/merge_data.py
    result_qwen3_debug/integrate_csv.py
    result_deepseek_stride1/merge_models.py

Each input CSV is expected to have columns
    layer, intrinsic_dimension, volume, tokens_used, stride

The output CSV adds ``model`` and ``relative_layer = layer / max(layer)``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from reasoning_manifolds.utils import configure_logging


SIZE_PATTERN = re.compile(r"(\d+\.?\d*)\s*B", re.IGNORECASE)


def parse_size(model: str) -> float:
    match = SIZE_PATTERN.search(model)
    return float(match.group(1)) if match else 0.0


def collect_csvs(root: Path, glob_pattern: str) -> list[tuple[str, Path]]:
    """Return ``(model_name, csv_path)`` pairs by matching ``glob_pattern``."""
    out: list[tuple[str, Path]] = []
    for path in sorted(root.rglob(glob_pattern)):
        # take the parent directory name as the model id, stripped of the
        # HuggingFace ``models--owner--`` prefix used by the hub cache
        model = path.parent.name
        if model.startswith("models--"):
            model = model.split("--", 2)[-1]
        out.append((model, path))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="merge per-model layer-wise metrics CSVs")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--glob", default="*_metrics.csv")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    configure_logging()

    pairs = collect_csvs(args.root, args.glob)
    if not pairs:
        raise SystemExit(f"no files matched {args.root}/**/{args.glob}")

    frames = []
    for model, path in pairs:
        df = pd.read_csv(path)
        for col in ("layer", "intrinsic_dimension", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["layer"])
        df["relative_layer"] = df["layer"] / df["layer"].max()
        df["model"] = model
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["__size"] = merged["model"].map(parse_size)
    merged = merged.sort_values(["__size", "model", "layer"]).drop(columns="__size")

    cols = ["model", "layer", "relative_layer", "intrinsic_dimension", "volume"]
    extra = [c for c in merged.columns if c not in cols]
    merged = merged[cols + extra]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"wrote {len(merged)} rows from {len(pairs)} models -> {args.output}")


if __name__ == "__main__":
    main()
