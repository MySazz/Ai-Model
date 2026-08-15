"""Retrieval of validated teacher examples for the agent loop.

Mirrors the Colab baseline recipe that produced the strongest untrained result
(Qwen3-4B + 2 retrieved validated examples: 10/12 with zero critical failures,
per `training/results/qwen3-4b-policy-retrieval-v1.json`): token-Jaccard
ranking of the user prompt against candidate prompts, top-k exemplar
user/assistant pairs injected before the real prompt.

Sources are the frozen, license-approved validator curricula (v2 + v3). Only
records flagged `validator_backed` are eligible. The module is stdlib-only and
deterministic; no network and no GPU are required.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCES = (
    "datasets/candidates/validator-curriculum-v2.jsonl",
    "datasets/candidates/validator-curriculum-v3.jsonl",
)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "that", "the", "to", "was", "with",
        "you", "your", "i", "we", "not", "do", "does", "can", "write",
        # common function words that cause false-positive overlap
        "me", "my", "our", "us", "about", "any", "anything", "this", "these",
        "those", "there", "here", "will", "would", "should", "could", "have",
        "has", "had", "been", "being", "am", "were", "did", "done", "get",
        "make", "need", "want", "please", "also", "just", "very", "really",
        "so", "but", "if", "then", "than", "when", "which", "what", "how",
        "why", "all", "both", "each", "few", "more", "most", "other", "such",
        "no", "nor", "only", "own", "same", "too",
    }
)
_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.casefold())) - _STOPWORDS


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    answer: str
    concept: str
    category: str


class ExampleBank:
    """Loads validated exemplars and ranks them against a user prompt."""

    def __init__(self, root: Path, sources: tuple[str, ...] = DEFAULT_SOURCES) -> None:
        self._examples: list[Example] = []
        for source in sources:
            path = root / source
            if not path.exists():
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                metadata = row.get("metadata") or {}
                if not metadata.get("validator_backed"):
                    continue
                messages = row.get("messages") or []
                if len(messages) < 2:
                    continue
                self._examples.append(
                    Example(
                        id=str(row.get("id", f"{path.stem}:{line_number}")),
                        prompt=str(messages[-2].get("content", "")),
                        answer=str(messages[-1].get("content", "")),
                        concept=str(metadata.get("behavior_concept", "")),
                        category=str(metadata.get("category", "")),
                    )
                )

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self._examples)

    def retrieve(self, prompt: str, k: int = 2) -> list[Example]:
        """Top-k validator-backed exemplars ranked by token overlap.

        Only exemplars with at least one overlapping token are returned, so an
        unrelated prompt (general chat) receives no examples rather than
        irrelevant ones. Ranking is deterministic: score desc, id asc.
        """
        if k <= 0 or not self._examples:
            return []
        wanted = _tokens(prompt)
        if not wanted:
            return []
        ranked: list[tuple[float, Example]] = []
        for example in self._examples:
            candidate = _tokens(example.prompt)
            union = len(wanted | candidate)
            if union == 0:
                continue
            score = len(wanted & candidate) / union
            if score > 0:
                ranked.append((score, example))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [example for _, example in ranked[:k]]
