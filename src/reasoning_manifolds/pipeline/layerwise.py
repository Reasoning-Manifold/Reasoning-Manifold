"""Layer-wise extraction script: dump per-sample, per-layer hidden states.

This is the analogue of ``Layer-ID/hs_layer.py`` and ``hs_layer_gemma.py``,
needed to reproduce the per-layer ID/V curves in Fig 1 and Fig 3B.

Run separately from the multi-repeat pipeline because each sample writes a
``[num_layers, num_tokens, hidden]`` ``.pt`` file.

Usage::

    python -m reasoning_manifolds.pipeline.layerwise \\
        --model Qwen/Qwen3-8B \\
        --dataset data/mmlu_other/sub_other0.jsonl \\
        --config qwen3 \\
        --output-dir results/layerwise/

Pair with ``scripts/compute_layerwise_metrics.py`` to turn the dumped ``.pt``
files into per-layer ID/V CSVs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from reasoning_manifolds.data import iter_jsonl
from reasoning_manifolds.extract import HiddenStateCollector, all_layer_ids, format_chat_input
from reasoning_manifolds.models import is_gemma, load_model_and_tokenizer
from reasoning_manifolds.prompts import format_mmlu_prompt, get_decoding_params
from reasoning_manifolds.utils import configure_logging, model_short_name, set_seed

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sample layer-wise hidden state extraction")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--config", required=True, choices=["qwen3", "qwen2.5", "deepseek", "gemma3", "greedy"]
    )
    p.add_argument("--output-dir", default="./results/layerwise")
    p.add_argument("--max-new-tokens", type=int, default=5000, help="Layer-ID experiments use 5000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(prefix="layerwise")
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model, tp_size=1)
    is_gemma_model = is_gemma(args.model)
    layer_ids = all_layer_ids(model)
    decoding = get_decoding_params(args.config)

    base = Path(args.output_dir) / model_short_name(args.model) / Path(args.dataset).stem
    states_dir = base / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    results_path = base / "results.jsonl"

    logger.info("output: %s", base)
    n = 0
    with results_path.open("w", encoding="utf-8") as out:
        for sample in tqdm(iter_jsonl(args.dataset), desc="samples"):
            if args.max_samples is not None and n >= args.max_samples:
                break
            prompt = format_mmlu_prompt(sample.question, sample.choices)
            tokens = format_chat_input(tokenizer, prompt, is_gemma=is_gemma_model)
            input_ids = tokens["input_ids"].to(model.device)
            attn = tokens["attention_mask"].to(model.device)
            prompt_len = input_ids.shape[-1]

            with HiddenStateCollector(model, layer_ids) as collector:
                with torch.no_grad():
                    try:
                        gen = model.generate(
                            input_ids,
                            attention_mask=attn,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            **decoding,
                        )
                    except Exception as exc:
                        logger.error("generate failed for sample %s: %s", sample.id, exc)
                        continue

                stacked = []
                ok = True
                for layer_id in layer_ids:
                    states = collector.layer_hs[layer_id][1:]
                    if not states:
                        ok = False
                        break
                    stacked.append(torch.cat(states, dim=0))
                if not ok:
                    continue
                final = torch.stack(stacked, dim=0)  # [layers, tokens, hidden]

            torch.save(final, states_dir / f"{sample.id}.pt")
            text = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            out.write(
                json.dumps(
                    {
                        "id": sample.id,
                        "problem": prompt,
                        "ground_truth": sample.answer,
                        "response": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

            n += 1
            if n % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    logger.info("wrote %d samples to %s", n, base)


if __name__ == "__main__":
    main()
