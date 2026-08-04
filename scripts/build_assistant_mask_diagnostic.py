#!/usr/bin/env python3
"""Select a small balanced authored corpus for the assistant-mask diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("datasets/candidates/capability-v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/candidates/assistant-mask-diagnostic-v1.jsonl"))
    parser.add_argument("--records", type=int, default=120)
    args = parser.parse_args()
    authored = []
    for line in args.source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"].startswith("capability-v2-authored-"):
            clone = dict(row)
            clone["id"] = row["id"].replace("capability-v2-authored-", "assistant-mask-diagnostic-")
            authored.append(clone)
            if len(authored) == args.records:
                break
    if len(authored) != args.records:
        raise SystemExit(f"Needed {args.records} authored records; found {len(authored)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in authored), encoding="utf-8")
    print(f"Wrote {len(authored)} diagnostic records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
