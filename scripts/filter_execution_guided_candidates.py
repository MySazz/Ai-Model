#!/usr/bin/env python3
"""Filter generated candidates into a validator-backed SFT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_agent.candidate_filtering import filter_candidates  # noqa: E402
from hybrid_agent.evaluation import load_jsonl_objects  # noqa: E402
from hybrid_agent.executable_evaluation import run_python_checks  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-per-task", type=int, default=2)
    args = parser.parse_args()
    tasks = load_jsonl_objects(args.tasks)
    candidates = load_jsonl_objects(args.candidates)
    accepted, report = filter_candidates(
        tasks, candidates, executable_judge=run_python_checks,
        minimum_per_task=args.minimum_per_task,
    )
    report["tasks_sha256"] = digest(args.tasks)
    report["candidates_sha256"] = digest(args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": len(accepted), "input": len(candidates),
        "ready_for_training": report["ready_for_training"],
        "output": str(args.output), "report": str(args.report),
    }))
    return 0 if report["ready_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
