#!/usr/bin/env python3
"""Combine diverse and concise authored curricula for transfer diagnosis."""

from __future__ import annotations

import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    diverse = [row for row in read(Path("datasets/candidates/capability-v2.jsonl"))
               if row["id"].startswith("capability-v2-authored-")]
    concise = read(Path("datasets/candidates/assistant-mask-diagnostic-v2.jsonl"))
    rows = []
    for source, prefix in ((diverse, "diverse"), (concise, "concise")):
        for index, row in enumerate(source):
            clone = dict(row)
            clone["id"] = f"generalization-diagnostic-v1-{prefix}-{index:06d}"
            rows.append(clone)
    rows.sort(key=lambda row: row["id"])
    output = Path("datasets/candidates/generalization-diagnostic-v1.jsonl")
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} records: diverse={len(diverse)} concise={len(concise)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
