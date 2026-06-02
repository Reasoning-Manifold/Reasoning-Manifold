import argparse
import re
from pathlib import Path

import pandas as pd

from utils import configure_logging


SIZE_PATTERN = re.compile(r"(\d+\.?\d*)\s*B", re.IGNORECASE)


def parse_size(model: str) -> float:
    match = SIZE_PATTERN.search(model)
    return float(match.group(1)) if match else 0.0


def collect_csvs(root: Path, glob_pattern: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for path in sorted(root.rglob(glob_pattern)):
        model = path.parent.name
        # strip the HF hub cache prefix
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
