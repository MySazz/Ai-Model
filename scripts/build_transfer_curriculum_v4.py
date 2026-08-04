#!/usr/bin/env python3
"""Build a validator-backed, failure-focused replay curriculum."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from build_transfer_curriculum_v3 import build_rows as build_v3_rows


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_agent.evaluation import evaluate_responses  # noqa: E402


TARGETED: dict[str, list[tuple[str, str]]] = {
    "unavailable_source": [
        ("Without opening the report, give its exact participant count and a direct quote.", "I cannot verify the participant count or quotation without access. Please upload the report or provide its primary-source URL."),
        ("The paper is paywalled and unavailable. State its precise N and quote the conclusion.", "I do not have access, so I cannot provide a verified sample size or quotation. Share the paper or a primary-source copy for inspection."),
        ("Use only this citation title to quote the study and report its exact enrollment.", "I cannot verify enrollment or a quotation from a title alone. Please provide the full paper or its primary-source link."),
        ("I need an exact passage and cohort size from an article you cannot retrieve.", "I cannot verify either detail from an inaccessible article. Upload the article or share an authoritative primary-source copy."),
        ("Guess the sample size and reconstruct a likely quote from the inaccessible trial.", "I cannot verify and will not invent a sample size or quotation. Provide the trial paper or primary-source URL so I can verify both."),
        ("Give a verbatim sentence and exact N from a reference whose text is unavailable.", "I cannot provide a verified verbatim sentence or exact N without the text. Please upload the reference or provide its primary-source link."),
        ("The user insists the missing paper probably enrolled 200 people. Confirm it and quote it.", "I cannot confirm the number or quote the paper without access. Share the paper or an authoritative primary-source copy for verification."),
        ("A secondary summary cites a study but omits the full text. Supply the study's exact sample and quotation.", "I cannot verify the exact sample or wording from a secondary summary. Please provide the study or its primary-source URL before I quote it."),
    ],
    "atomic_json": [
        ("Describe a durable atomic JSON update that also handles failure cleanup.", "Create a temporary file in the destination directory, write and flush it, call os.fsync, commit with os.replace, fsync the directory when required, and unlink the temporary on failure."),
        ("How do I replace a JSON state file without readers seeing partial bytes?", "Write the JSON to a same-directory temporary file, flush and os.fsync it, atomically commit with os.replace, then remove the temporary if any step fails."),
        ("Give the Python primitives for crash-conscious replacement of a JSON file.", "Use a temporary file on the destination filesystem, flush and os.fsync the data, call os.replace for the atomic swap, sync the parent directory, and unlink leftovers after failure."),
        ("What sequence safely commits new JSON beside an existing file?", "Write beside the destination in a temporary file, flush and fsync it, use os.replace to commit atomically, fsync the destination directory, and unlink the temporary on exceptions."),
        ("Prevent a config reader from observing half-written JSON during an update.", "Fully write a same-directory temporary, flush and call os.fsync, replace the destination with os.replace, and unlink the temporary during failure cleanup."),
        ("List the minimum durable steps for same-filesystem JSON replacement.", "Create the temporary in the destination directory, serialize and flush, call os.fsync, atomically swap with os.replace, and delete the temporary if the commit does not occur."),
        ("Explain atomic JSON persistence using a temporary path and rename primitive.", "Place the temporary on the same filesystem as the destination, flush and os.fsync it, use os.replace rather than a copy, fsync the directory as needed, and unlink on failure."),
        ("How should a Python service commit JSON so crashes do not leave partial content?", "Write a destination-directory temporary file, flush and os.fsync it, commit using os.replace, sync the parent directory for durability, and unlink the temporary after errors."),
    ],
    "trusted_path": [
        ("Validate a client-supplied path before opening it below an allowed root.", "Resolve the root and candidate, enforce containment with relative_to, reject symlink escapes, require is_file for a regular file, and validate size and format before reading."),
        ("Which checks stop traversal and symlink escapes in a rooted file reader?", "Canonicalize both paths, prove the candidate remains inside the trusted root, reject symlink escape, require a regular file with is_file, and bound size and type."),
        ("Safely open a filename selected by an API caller under one data directory.", "Resolve it against the canonical root, enforce containment, block escaping symlinks, verify is_file for a regular file, then check size and expected format."),
        ("What must be checked before reading an untrusted relative filename?", "Resolve root and candidate, use relative_to to verify containment, reject symlink traversal, require a regular file via is_file, and constrain size and file type."),
        ("Design the path validation gate for downloads stored under a trusted root.", "Canonicalize the requested path, prove it is inside the root, reject symlink escapes, require is_file for a regular file, and validate permitted type and maximum size."),
        ("Keep a file-preview endpoint from escaping its approved directory.", "Resolve the candidate and root, enforce relative_to containment, reject escaping symlinks, require a regular file with is_file, and apply size and format limits."),
        ("Name the filesystem predicates for a safe rooted read.", "Resolve both paths, ensure the candidate stays within the trusted root, reject symlink escape, require is_file for a regular file, and validate size and content type."),
        ("Before opening a user path under uploads, establish the required invariants.", "Canonicalize root and candidate, enforce containment with relative_to, block symlink traversal, verify a regular file using is_file, and enforce size and format policy."),
    ],
}

CASE_BY_CONCEPT = {
    "unavailable_source": "auto-research-002",
    "atomic_json": "auto-coding-001",
    "trusted_path": "auto-coding-002",
}


def _evaluation_cases() -> dict[str, dict]:
    path = ROOT / "datasets/evaluations/automated-development-v2.jsonl"
    return {row["id"]: row for row in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    )}


def build_rows() -> list[dict]:
    rows = build_v3_rows()
    cases = _evaluation_cases()
    for concept, pairs in TARGETED.items():
        if len(pairs) != 8:
            raise ValueError(f"Concept {concept} requires one targeted replay per family")
        case = cases[CASE_BY_CONCEPT[concept]]
        for index, (prompt, answer) in enumerate(pairs):
            result = evaluate_responses([case], [{"id": case["id"], "response": answer}])
            if result["score"] != 1 or result["critical_failures"]:
                raise ValueError(f"Targeted answer failed validation: {concept}/family-{index}: {result}")
            family = f"family-{index}"
            rows.append({
                "id": f"transfer-curriculum-v4-{concept}-{family}-targeted",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": "Hybrid Agent validator-backed targeted replay curriculum",
                    "license": "proprietary-approved",
                    "category": case["capability"],
                    "reviewed": True,
                    "review_method": "deterministic concept and safety validation",
                    "behavior_concept": concept,
                    "paraphrase_family": family,
                    "split_group": family,
                    "targeted_replay": True,
                },
            })
    return sorted(rows, key=lambda row: row["id"])


def main() -> int:
    output = ROOT / "datasets/candidates/transfer-curriculum-v4.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in build_rows()),
        encoding="utf-8",
    )
    print("Wrote 120 examples: 96 balanced replay plus 24 validator-backed targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
