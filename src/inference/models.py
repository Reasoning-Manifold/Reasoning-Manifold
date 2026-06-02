import logging
from typing import Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


def load_model_and_tokenizer(
    model_id: str,
    *,
    tp_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    logger.info("Loading model: %s (tp=%d)", model_id, tp_size)

    if is_gemma(model_id):
        # Gemma3's chat template differs from AutoModel default
        from transformers import Gemma3ForCausalLM

        model_cls = Gemma3ForCausalLM
        kwargs: dict = {"torch_dtype": dtype}
    else:
        model_cls = AutoModelForCausalLM
        kwargs = {"torch_dtype": dtype, "trust_remote_code": True}

    if tp_size <= 1:
        model = model_cls.from_pretrained(model_id, **kwargs).to("cuda:0").eval()
    else:
        try:
            import tensor_parallel as tp_lib

            model = model_cls.from_pretrained(model_id, **kwargs)
            model = tp_lib.tensor_parallel(model, [f"cuda:{i}" for i in range(tp_size)]).eval()
        except ImportError:
            logger.warning("tensor_parallel missing, falling back to device_map='auto'")
            model = model_cls.from_pretrained(model_id, device_map="auto", **kwargs).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return model, tokenizer


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot locate decoder layers on {type(model).__name__}")


def num_decoder_layers(model: torch.nn.Module) -> int:
    return len(get_decoder_layers(model))


def vocab_embedding_matrix(model: torch.nn.Module) -> np.ndarray:
    return (
        model.get_input_embeddings()
        .weight.detach()
        .to(dtype=torch.float32, device="cpu")
        .numpy()
        .astype(np.float64)
    )
