"""Hidden-state extraction during autoregressive generation.

The paper takes the *last-token* hidden state at every transformer layer for
every generation step (§Methods, Eq. 2-3). Two collection modes are needed:

* ``"all_layers"`` — used by Layer-ID experiments to study layer-wise ID/V
  curves (Fig 1, Fig 3B).
* ``"last_layer"`` — used by the multi-repeat aggregator to compute
  ``D_stim`` and ``V`` at the final layer for ℋ.

This module replaces the duplicated hook code in ``intelligence/worker.py``,
``Layer-ID/hs_layer.py``, ``Layer-ID/hs_layer_gemma.py`` and
``results/gpqa.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch

from reasoning_manifolds.models import get_decoder_layers, num_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class HiddenStateCollector:
    """Capture the last-token hidden state at every selected layer per step.

    For each ``forward`` of the model, the hook reads ``output[:, -1, :]``
    and stores it on CPU in float32. After ``model.generate`` finishes,
    ``layer_hs[layer_id]`` is a list with one tensor per step (shape
    ``[batch, hidden]``).
    """

    model: torch.nn.Module
    layer_ids: Sequence[int]
    layer_hs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    handles: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.layer_hs = {layer_id: [] for layer_id in self.layer_ids}

    def _hook(self, layer_id: int):
        def fn(module, inputs, output):  # noqa: ARG001
            hidden = output[0] if isinstance(output, tuple) else output
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
        """Stack a layer's per-step states into ``[steps, batch, hidden]``.

        Args:
            drop_prefill: skip step 0 (the prefill forward), matching the
                convention in ``Layer-ID/hs_layer.py``.
        """
        states = self.layer_hs[layer_id]
        if drop_prefill:
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
    """Apply the model's chat template and return tokenizer outputs.

    Gemma3 uses a nested message structure; everything else uses the standard
    HF chat template.
    """
    if is_gemma:
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
