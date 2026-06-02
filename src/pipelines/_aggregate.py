import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.metrics import (
    EPSILON_DEFAULT,
    information_volume,
    intrinsic_dimension,
    reasoning_health,
)

logger = logging.getLogger(__name__)


@dataclass
class RepeatResult:
    repeat_id: int
    D_stim: float
    V: float
    H: float


def load_d_world(temp_dir: Path, dp: int) -> float:
    values: list[float] = []
    for worker_id in range(dp):
        path = temp_dir / f"worker_{worker_id}" / "d_world.json"
        if path.exists():
            values.append(json.loads(path.read_text())["D_world"])
    if not values:
        raise FileNotFoundError(f"no d_world.json files under {temp_dir}")
    return float(np.mean(values))


def load_hidden_states(temp_dir: Path, repeat_id: int, dp: int) -> torch.Tensor:
    for worker_id in range(dp):
        path = temp_dir / f"worker_{worker_id}" / f"hs_r{repeat_id}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            return torch.cat([s["states"] for s in payload["hidden_states"]], dim=0)
    raise FileNotFoundError(f"no hs_r{repeat_id}.pt under {temp_dir}")


def aggregate(
    *,
    temp_dir: Path,
    output_dir: Path,
    total_repeats: int,
    dp: int,
    epsilon: float = EPSILON_DEFAULT,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    d_world = load_d_world(temp_dir, dp)
    logger.info("D_world = %.4f", d_world)

    repeat_results: list[RepeatResult] = []
    for repeat_id in range(total_repeats):
        states = load_hidden_states(temp_dir, repeat_id, dp).numpy().astype(np.float64)
        d_stim = intrinsic_dimension(states)
        volume = information_volume(states)
        h = reasoning_health(d_world, d_stim, volume, epsilon=epsilon)
        repeat_results.append(RepeatResult(repeat_id, d_stim, volume, h))
        logger.info("repeat %d: D_stim=%.4f V=%.4f H=%.4f", repeat_id, d_stim, volume, h)

    summary = _summarise(d_world, repeat_results, epsilon)
    report = {
        "config": config or {},
        "summary": summary,
        "repeats": [asdict(r) for r in repeat_results],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(report), encoding="utf-8")
    _merge_predictions(temp_dir, output_dir, total_repeats, dp)
    return report


def _summarise(d_world: float, repeats: list[RepeatResult], epsilon: float) -> dict[str, Any]:
    def arr(key: str) -> np.ndarray:
        return np.array([getattr(r, key) for r in repeats], dtype=np.float64)

    return {
        "D_world": d_world,
        "epsilon": epsilon,
        "D_stim": _stats(arr("D_stim")),
        "V": _stats(arr("V")),
        "H": _stats(arr("H")),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def _render_report(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Reasoning Health Report",
        "",
        f"- model: `{report['config'].get('model', '?')}`",
        f"- dataset: `{report['config'].get('dataset', '?')}`",
        f"- repeats: {len(report['repeats'])}",
        f"- D_world: **{s['D_world']:.4f}**",
        f"- epsilon: {s['epsilon']}",
        "",
        "| metric | mean | std | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in ("D_stim", "V", "H"):
        st = s[key]
        lines.append(
            f"| {key} | {st['mean']:.4f} | {st['std']:.4f} | "
            f"{st['min']:.4f} | {st['max']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _merge_predictions(temp_dir: Path, output_dir: Path, total_repeats: int, dp: int) -> None:
    dest = output_dir / "predictions"
    dest.mkdir(parents=True, exist_ok=True)
    for repeat_id in range(total_repeats):
        for worker_id in range(dp):
            src = temp_dir / f"worker_{worker_id}" / f"pred_r{repeat_id}.jsonl"
            if src.exists():
                shutil.copy(src, dest / f"repeat_{repeat_id:02d}.jsonl")
                break
