import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch

from inference.models import get_decoder_layers, num_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class HiddenStateCollector:
    model: torch.nn.Module
    layer_ids: Sequence[int]
    layer_hs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    handles: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.layer_hs = {layer_id: [] for layer_id in self.layer_ids}

    def _hook(self, layer_id: int):
        def fn(module, inputs, output):  # noqa: ARG001
            hidden = output[0] if isinstance(output, tuple) else output
            # last-token hidden state on CPU, float32
            self.layer_hs[layer_id].append(hidden[:, -1, :].detach().cpu().to(torch.float32))

        return fn

    def __enter__(self) -> "HiddenStateCollector":
        layers = get_decoder_layers(self.model)
        for layer_id in self.layer_ids:
            self.handles.append(layers[layer_id].register_forward_hook(self._hook(layer_id)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ARG002
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        for layer_id in self.layer_ids:
            self.layer_hs[layer_id] = []

    def as_tensor(self, layer_id: int, *, drop_prefill: bool = True) -> torch.Tensor:
        states = self.layer_hs[layer_id]
        if drop_prefill:
            # step 0 is the prefill forward
            states = states[1:]
        if not states:
            return torch.empty(0)
        return torch.stack(states, dim=0)


def all_layer_ids(model: torch.nn.Module) -> list[int]:
    return list(range(num_decoder_layers(model)))


def format_chat_input(
    tokenizer,
    prompt: str,
    *,
    is_gemma: bool = False,
) -> dict[str, torch.Tensor]:
    if is_gemma:
        # Gemma3 uses a nested message structure
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out.get("attention_mask", torch.ones_like(out["input_ids"])),
        }

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
