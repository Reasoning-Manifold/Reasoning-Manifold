import argparse
import json
import logging
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

from core.data import load_jsonl
from core.metrics import intrinsic_dimension
from inference.decoding import get_decoding_params
from inference.extract import HiddenStateCollector, format_chat_input
from inference.models import (
    is_gemma,
    load_model_and_tokenizer,
    num_decoder_layers,
    vocab_embedding_matrix,
)
from inference.prompts import format_mmlu_prompt
from utils import configure_logging, set_seed

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning Manifolds — worker")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--config", required=True, choices=["qwen3", "qwen2.5", "deepseek", "gemma3", "greedy"]
    )
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--repeat-start", type=int, required=True)
    p.add_argument("--repeat-end", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def run_repeat(
    *,
    model,
    tokenizer,
    samples,
    last_layer: int,
    repeat_id: int,
    output_dir: Path,
    decoding_params: dict,
    max_new_tokens: int,
    is_gemma_model: bool,
) -> None:
    hidden_states_per_sample: list[dict] = []
    predictions: list[dict] = []

    with HiddenStateCollector(model, [last_layer]) as collector:
        for sample in tqdm(samples, desc=f"repeat {repeat_id}"):
            prompt = format_mmlu_prompt(sample.question, sample.choices)
            tokens = format_chat_input(tokenizer, prompt, is_gemma=is_gemma_model)
            input_ids = tokens["input_ids"].to(model.device)
            attention_mask = tokens["attention_mask"].to(model.device)
            prompt_len = input_ids.shape[-1]

            collector.clear()
            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    **decoding_params,
                )

            text = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            states = collector.as_tensor(last_layer, drop_prefill=True)
            if states.numel() == 0:
                continue

            hidden_states_per_sample.append({"sample_id": sample.id, "states": states.squeeze(1)})
            predictions.append(
                {
                    "id": sample.id,
                    "question": sample.question,
                    "ground_truth": sample.answer,
                    "prediction": text,
                }
            )

    torch.save(
        {
            "repeat_id": repeat_id,
            "hidden_states": hidden_states_per_sample,
            "num_samples": len(samples),
        },
        output_dir / f"hs_r{repeat_id}.pt",
    )
    pred_path = output_dir / f"pred_r{repeat_id}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for entry in predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("repeat %d done: %d samples", repeat_id, len(predictions))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(prefix=f"worker {args.worker_id}")

    output_dir = Path(args.output_dir) / f"worker_{args.worker_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, tp_size=args.tp)
    is_gemma_model = is_gemma(args.model)
    last_layer = num_decoder_layers(model) - 1
    decoding_params = get_decoding_params(args.config)

    logger.info("computing D_world")
    d_world = intrinsic_dimension(vocab_embedding_matrix(model))
    (output_dir / "d_world.json").write_text(json.dumps({"D_world": d_world}))

    samples = load_jsonl(args.dataset)
    logger.info("loaded %d samples", len(samples))

    for repeat_id in range(args.repeat_start, args.repeat_end):
        set_seed(args.seed + repeat_id)
        run_repeat(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            last_layer=last_layer,
            repeat_id=repeat_id,
            output_dir=output_dir,
            decoding_params=decoding_params,
            max_new_tokens=args.max_new_tokens,
            is_gemma_model=is_gemma_model,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
