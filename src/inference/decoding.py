from typing import Any

# decoding params mirror the paper's per-family settings
DECODING_PARAMS: dict[str, dict[str, Any]] = {
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "do_sample": True},
    "qwen2.5": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "do_sample": True},
    "deepseek": {"temperature": 0.6, "top_p": 0.95, "do_sample": True},
    "gemma3": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "do_sample": True},
    "greedy": {"do_sample": False},
}


def get_decoding_params(family: str | None) -> dict[str, Any]:
    if family is None:
        return DECODING_PARAMS["greedy"]
    if family not in DECODING_PARAMS:
        raise KeyError(
            f"Unknown decoding family {family!r}. Choose from {sorted(DECODING_PARAMS)}."
        )
    return DECODING_PARAMS[family]
