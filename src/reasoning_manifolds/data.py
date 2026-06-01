"""Stimulus / dataset loaders.

The paper draws inference-time stimuli from MMLU-Other, partitioned into
13 disjoint question types for the stimulus-expansion experiment. Other
benchmarks (AIME'25, GPQA-Diamond) follow the same JSONL schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


@dataclass
class Stimulus:
    """One stimulus / question with optional choices and ground truth."""

    id: str | int
    question: str
    choices: list[str] | None = None
    answer: str | int | None = None


_QUESTION_KEYS = ("question", "problem", "Question")
_ANSWER_KEYS = ("answer", "solution", "ground_truth", "Correct Answer")


def _pick(record: dict, keys: Iterable[str]):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[Stimulus]:
    """Load a JSONL stimulus file.

    Accepts the union of field names used across MMLU/AIME/GPQA dumps in
    this project: ``question`` | ``problem``, ``answer`` | ``solution`` |
    ``ground_truth``, optional ``choices``.
    """
    path = Path(path)
    out: list[Stimulus] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("%s:%d  invalid JSON, skipping", path, idx)
                continue
            question = _pick(record, _QUESTION_KEYS)
            if not question:
                continue
            out.append(
                Stimulus(
                    id=record.get("id", record.get("ID", idx)),
                    question=str(question),
                    choices=record.get("choices"),
                    answer=_pick(record, _ANSWER_KEYS),
                )
            )
            if max_samples is not None and len(out) >= max_samples:
                break
    return out


def iter_jsonl(path: str | Path) -> Iterator[Stimulus]:
    """Iterate stimuli without holding the whole list in memory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            question = _pick(record, _QUESTION_KEYS)
            if not question:
                continue
            yield Stimulus(
                id=record.get("id", record.get("ID", idx)),
                question=str(question),
                choices=record.get("choices"),
                answer=_pick(record, _ANSWER_KEYS),
            )
