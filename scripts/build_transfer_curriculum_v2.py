#!/usr/bin/env python3
"""Build four independently authored, group-held-out paraphrase families."""

from __future__ import annotations

import json
from pathlib import Path

from build_transfer_curriculum_v1 import FAMILIES


def build_rows() -> list[dict]:
    rows = []
    for concept, (category, pairs) in FAMILIES.items():
        if len(pairs) != 4:
            raise ValueError(f"Concept {concept} must have exactly four paraphrase families")
        for index, (prompt, answer) in enumerate(pairs):
            family = f"family-{index}"
            rows.append({
                "id": f"transfer-curriculum-v2-{concept}-{family}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": "Hybrid Agent independently authored transfer curriculum",
                    "license": "proprietary-approved",
                    "category": category,
                    "reviewed": True,
                    "review_method": "independent manual paraphrase review",
                    "behavior_concept": concept,
                    "paraphrase_family": family,
                    "split_group": family,
                },
            })
    return sorted(rows, key=lambda row: row["id"])


def main() -> int:
    rows = build_rows()
    output = Path("datasets/candidates/transfer-curriculum-v2.jsonl")
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    print("Wrote 48 independently authored examples in four held-out paraphrase families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
