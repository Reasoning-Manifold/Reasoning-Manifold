"""Top-level launcher: spawn workers, then aggregate.

Entry point for the ``reasoning-manifolds`` console script.

Usage::

    reasoning-manifolds run \\
        --model Qwen/Qwen3-8B \\
        --dataset data/mmlu_other/sub_other0.jsonl \\
        --config qwen3 \\
        --tp 1 --dp 1 --repeats 1 \\
        --output-dir results/qwen3-8b/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch

from reasoning_manifolds.pipeline.aggregator import aggregate
from reasoning_manifolds.utils import (
    allocate_gpus,
    allocate_repeats,
    configure_logging,
    ensure_dir,
    model_short_name,
)
from reasoning_manifolds.metrics import EPSILON_DEFAULT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reasoning-manifolds")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="extract hidden states and compute D_world / D_stim / V / H")
    run.add_argument("--model", required=True, help="HF model id or local path")
    run.add_argument("--dataset", required=True, help="path to JSONL stimuli")
    run.add_argument(
        "--config",
        required=True,
        choices=["qwen3", "qwen2.5", "deepseek", "gemma3", "greedy"],
        help="decoding family (matches paper)",
    )
    run.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    run.add_argument("--dp", type=int, default=1, help="data-parallel size (workers)")
    run.add_argument("--repeats", type=int, default=1, help="number of generation repeats")
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--max-new-tokens", type=int, default=15000, help="paper uses 15000")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help="ε in H = log(D_world)·V/exp(ε·D_stim)",
    )
    run.add_argument("--output-dir", default="./results")
    run.add_argument(
        "--keep-temp", action="store_true", help="do not delete the worker temp directory"
    )
    run.add_argument(
        "--no-aggregate", action="store_true", help="run workers only; skip the aggregation step"
    )

    agg = sub.add_parser("aggregate", help="aggregate existing worker outputs")
    agg.add_argument("--temp-dir", required=True)
    agg.add_argument("--output-dir", required=True)
    agg.add_argument("--repeats", type=int, required=True)
    agg.add_argument("--dp", type=int, default=1)
    agg.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT)

    return p.parse_args(argv)


def _layout(output_dir: str, model: str, dataset: str) -> dict[str, Path]:
    base = Path(output_dir) / model_short_name(model) / Path(dataset).stem
    return {"base": base, "temp": base / "temp"}


def _spawn_worker(
    *,
    worker_id: int,
    gpus: list[int],
    repeat_start: int,
    repeat_end: int,
    args: argparse.Namespace,
    temp_dir: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    cmd = [
        sys.executable,
        "-m",
        "reasoning_manifolds.pipeline.worker",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--config",
        args.config,
        "--tp",
        str(args.tp),
        "--worker-id",
        str(worker_id),
        "--repeat-start",
        str(repeat_start),
        "--repeat-end",
        str(repeat_end),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(temp_dir),
    ]
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _stream(process: subprocess.Popen, prefix: str) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"[{prefix}] {line}")


def cmd_run(args: argparse.Namespace) -> int:
    configure_logging(prefix="launcher")

    if args.tp * args.dp != torch.cuda.device_count():
        raise SystemExit(
            f"TP({args.tp})·DP({args.dp})={args.tp*args.dp} != "
            f"{torch.cuda.device_count()} visible GPUs"
        )
    if args.repeats % args.dp:
        raise SystemExit(f"--repeats {args.repeats} must be divisible by --dp {args.dp}")

    paths = _layout(args.output_dir, args.model, args.dataset)
    ensure_dir(paths["base"])
    ensure_dir(paths["temp"])

    gpu_groups = allocate_gpus(args.tp, args.dp)
    repeat_groups = allocate_repeats(args.repeats, args.dp)

    processes: list[subprocess.Popen] = []
    for worker_id, (gpus, repeats) in enumerate(zip(gpu_groups, repeat_groups)):
        proc = _spawn_worker(
            worker_id=worker_id,
            gpus=gpus,
            repeat_start=repeats[0],
            repeat_end=repeats[-1] + 1,
            args=args,
            temp_dir=paths["temp"],
        )
        processes.append(proc)

    threads = [
        threading.Thread(target=_stream, args=(p, f"worker {i}"), daemon=True)
        for i, p in enumerate(processes)
    ]
    for t in threads:
        t.start()
    return_codes = [p.wait() for p in processes]
    for t in threads:
        t.join(timeout=1)

    if any(rc != 0 for rc in return_codes):
        raise SystemExit(f"workers failed: return codes {return_codes}")

    if args.no_aggregate:
        return 0

    config = {
        "model": args.model,
        "dataset": args.dataset,
        "tp": args.tp,
        "dp": args.dp,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    aggregate(
        temp_dir=paths["temp"],
        output_dir=paths["base"],
        total_repeats=args.repeats,
        dp=args.dp,
        epsilon=args.epsilon,
        config=config,
    )

    if not args.keep_temp:
        shutil.rmtree(paths["temp"], ignore_errors=True)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    configure_logging(prefix="aggregate")
    aggregate(
        temp_dir=Path(args.temp_dir),
        output_dir=Path(args.output_dir),
        total_repeats=args.repeats,
        dp=args.dp,
        epsilon=args.epsilon,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
