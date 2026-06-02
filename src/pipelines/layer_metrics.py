import argparse
import csv
import glob
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from core.metrics import information_volume, intrinsic_dimension
from utils import configure_logging

logger = logging.getLogger(__name__)


def analyse(states_dir: Path, output_dir: Path, stride: int) -> None:
    pt_files = sorted(glob.glob(str(states_dir / "*.pt")))
    if not pt_files:
        raise SystemExit(f"no .pt files under {states_dir}")

    parts = states_dir.parts
    if len(parts) >= 3:
        model_name, dataset_name = parts[-3], parts[-2]
    else:
        model_name, dataset_name = "unknown", "unknown"
    base_name = f"{model_name}_{dataset_name}"

    first = torch.load(pt_files[0], map_location="cpu")
    num_layers, _, hidden_dim = first.shape

    layer_data: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    total_tokens = 0
    for path in tqdm(pt_files, desc="loading"):
        try:
            tensor = torch.load(path, map_location="cpu")
        except Exception as exc:
            logger.error("load %s failed: %s", path, exc)
            continue
        total_tokens += tensor.shape[1]
        for layer_idx in range(num_layers):
            layer_data[layer_idx].append(tensor[layer_idx])

    ids: list[float] = []
    vols: list[float] = []
    used: list[int] = []
    for layer_idx in tqdm(range(num_layers), desc="compute ID/V"):
        combined = torch.cat(layer_data[layer_idx], dim=0)
        sampled = combined[::stride].numpy().astype(np.float64)
        used.append(sampled.shape[0])
        ids.append(intrinsic_dimension(sampled.astype(np.float32)))
        vols.append(information_volume(sampled))

    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{base_name}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "intrinsic_dimension", "volume", "tokens_used", "stride"])
        for i, (id_v, vol_v, n) in enumerate(zip(ids, vols, used)):
            writer.writerow([i, id_v, vol_v, n, stride])

    json_path = out_dir / f"{base_name}_metrics.json"
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "num_samples": len(pt_files),
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "total_tokens": total_tokens,
                    "stride": stride,
                },
                "layer_results": [
                    {
                        "layer": i,
                        "intrinsic_dimension": None if np.isnan(id_v) else float(id_v),
                        "volume": None if (np.isnan(vol_v) or np.isinf(vol_v)) else float(vol_v),
                        "tokens_used": int(n),
                    }
                    for i, (id_v, vol_v, n) in enumerate(zip(ids, vols, used))
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="per-layer ID/V from saved .pt files")
    parser.add_argument("--states-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    configure_logging()
    analyse(args.states_dir, args.output_dir, args.stride)


if __name__ == "__main__":
    main()
