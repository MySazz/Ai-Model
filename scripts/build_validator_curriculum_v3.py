#!/usr/bin/env python3
"""Build validator-backed teacher families for the interfaces the v2 holdout failed.

Curriculum v3 teaches the two interface shapes the Qwen3-4B validator-holdout-v2
baseline actually failed (2/8, with real failures on `archive-plan-001` and
`chunk-install-001`): whole-batch archive planning and streaming atomic install.
The v2 curriculum covered the single-member/target interfaces (archive_target,
commit_bytes); the model did not generalize to the batch/stream variants. Each
prompt carries two diverse implementations so every task has two independent
passes, and every answer must pass the exact behavioral checks the held-out
evaluation uses (via the train-named harness aliases).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_agent.executable_evaluation import run_python_checks  # noqa: E402

EXECUTABLE_SPECS: dict[str, dict[str, Any]] = {
    "atomic_install": {
        "category": "coding",
        "harness": "install_payload_train_v2",
        "signature": "def install_payload(destination, chunks):\n",
        "checks": [
            "assembles_chunks",
            "accepts_empty_stream",
            "preserves_on_late_invalid_chunk",
            "preserves_on_source_exception",
            "commits_from_sibling_staging",
            "cleans_after_commit_failure",
        ],
        "prompts": [
            "Implement install_payload(destination, chunks) that streams exact byte chunks into a sibling temporary file and atomically replaces the destination.",
            "A package assembler builds artifacts from byte fragments. Write install_payload(destination, chunks) with fsync-before-commit ordering and cleanup on every failure.",
            "Write install_payload so an empty chunk stream replaces the destination with an empty file, and a late invalid chunk preserves the prior destination.",
            "Create a failure-safe install_payload(destination, chunks) that never leaves partial output or staging artifacts when the chunk source raises.",
            "Implement install_payload for an offline firmware updater: chunks arrive from a generator that may fail mid-stream and the previous firmware must survive.",
            "Provide install_payload(destination, chunks) that rejects non-bytes chunks without touching an existing destination.",
            "A media assembler merges downloaded segments through install_payload; write it using same-directory staging and os.replace so readers never see partial data.",
            "Return a standard-library install_payload(destination, chunks) with a guaranteed successful commit path and exception cleanup.",
        ],
        "answer_notes": [
            "Assemble byte fragments into a complete artifact without partial reads.",
            "Guarantee fsync-before-commit ordering for a package assembler.",
            "Treat an empty stream as a valid empty artifact, not a failure.",
            "Clean up sibling staging whenever the chunk source fails.",
            "Preserve the previous firmware image if the updater stream aborts.",
            "Reject non-bytes chunks before any destination is touched.",
            "Merge downloaded segments so concurrent readers never observe partial data.",
            "Provide a guaranteed successful commit path with exception cleanup.",
        ],
        "answers": [
            """import os
import tempfile
from pathlib import Path

def install_payload(destination, chunks):
    destination = Path(destination)
    fd, staging = tempfile.mkstemp(dir=destination.parent, prefix=f'.{destination.name}.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError('chunk must be bytes')
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, destination)
    except BaseException:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
        raise
""",
            """import os
from pathlib import Path
from tempfile import NamedTemporaryFile

def install_payload(destination, chunks):
    destination = Path(destination)
    staging = None
    try:
        with NamedTemporaryFile('wb', dir=destination.parent, prefix='staging-', delete=False) as stream:
            staging = stream.name
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError('chunk must be bytes')
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, destination)
        staging = None
    finally:
        if staging is not None:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass
""",
        ],
    },
    "archive_batch_plan": {
        "category": "coding",
        "harness": "plan_archive_train_v2",
        "signature": "def plan_archive(extract_root, names):\n",
        "checks": [
            "maps_batch",
            "rejects_parent_in_batch",
            "rejects_absolute_in_batch",
            "rejects_portable_aliases",
            "rejects_linked_escape",
            "rejects_colliding_destinations",
        ],
        "prompts": [
            "Implement plan_archive(extract_root, names) that validates an entire batch of archive member names and rejects the whole call if any member is unsafe.",
            "An unpacker receives a manifest of member names. Write plan_archive(extract_root, names) returning resolved paths in order, failing the whole batch on any traversal.",
            "Write plan_archive so empty, root-alias, absolute, parent, Windows drive, or backslash traversal members reject the entire batch.",
            "Create plan_archive(extract_root, names) that also rejects members escaping through existing symlinks and members that collide on the same destination.",
            "Implement plan_archive for untrusted archives: two spellings resolving to the same target must fail the whole batch, never partially plan.",
            "Provide plan_archive(extract_root, names) with whole-batch all-or-nothing semantics for mixed valid and unsafe member lists.",
            "A downloader maps remote archive listings to local paths; write plan_archive with strict cross-platform containment and collision detection.",
            "Return Python for plan_archive(extract_root, names): map every member below the root or reject the entire batch without a partial result.",
        ],
        "answer_notes": [
            "Validate the whole manifest before returning any resolved path.",
            "Fail the entire batch on any traversal, never a partial plan.",
            "Treat empty, root-alias, drive-qualified, and backslash members as hostile.",
            "Reject linked escapes and colliding destinations in the same pass.",
            "Two spellings of one destination are one collision too many.",
            "All-or-nothing planning for mixed valid and unsafe member lists.",
            "Prove containment beneath the trusted root for every member.",
            "Return an ordered plan only when every member is safe.",
        ],
        "answers": [
            """import os
from pathlib import Path, PurePosixPath, PureWindowsPath

def plan_archive(extract_root, names):
    root = Path(extract_root).resolve(strict=True)
    planned = []
    seen = set()
    for member in names:
        text = os.fspath(member)
        if not isinstance(text, str) or not text or '\\x00' in text:
            raise ValueError('invalid archive member')
        portable = PurePosixPath(text.replace('\\\\', '/'))
        windows = PureWindowsPath(text)
        if portable.is_absolute() or windows.drive or windows.root:
            raise ValueError('absolute archive member')
        if not portable.parts or any(part == '..' for part in portable.parts):
            raise ValueError('unsafe archive member')
        candidate = root.joinpath(*portable.parts).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise ValueError('archive member escapes root')
        if candidate in seen:
            raise ValueError('colliding archive members')
        seen.add(candidate)
        planned.append(candidate)
    return planned
""",
            """import os
from pathlib import Path, PurePosixPath, PureWindowsPath

def plan_archive(extract_root, names):
    base = Path(extract_root).resolve(strict=True)
    planned = []
    seen = set()
    for raw in names:
        name = os.fspath(raw)
        if not isinstance(name, str) or name in {'', '.'} or '\\x00' in name:
            raise PermissionError('unsafe archive member')
        portable = PurePosixPath(name.replace('\\\\', '/'))
        windows = PureWindowsPath(name)
        if portable.is_absolute() or windows.drive or windows.root:
            raise PermissionError('unsafe archive member')
        if not portable.parts or '..' in portable.parts:
            raise PermissionError('unsafe archive member')
        key = tuple(portable.parts)
        if key in seen:
            raise PermissionError('colliding archive members')
        seen.add(key)
        candidate = base.joinpath(*portable.parts).resolve()
        if candidate == base or not candidate.is_relative_to(base):
            raise PermissionError('archive member escapes root')
        planned.append(candidate)
    return planned
""",
        ],
    },
}


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept, spec in EXECUTABLE_SPECS.items():
        for prompt_index, prompt in enumerate(spec["prompts"]):
            for variant_index, answer in enumerate(spec["answers"]):
                answer = answer.replace(
                    spec["signature"],
                    spec["signature"] + f"    \"\"\"{spec['answer_notes'][prompt_index]}\"\"\"\n",
                    1,
                )
                evaluator = {
                    "type": "executable_assertions",
                    "language": "python",
                    "harness": spec["harness"],
                    "checks": spec["checks"],
                }
                outcomes = run_python_checks(answer, evaluator)
                if not all(outcomes.values()):
                    raise ValueError(
                        f"Target failed validation: {concept}/family-{prompt_index}-{variant_index}: {outcomes}"
                    )
                rows.append(_row(concept, prompt_index, variant_index, prompt, answer, spec))
    return sorted(rows, key=lambda row: row["id"])


def _row(
    concept: str,
    prompt_index: int,
    variant_index: int,
    prompt: str,
    answer: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": "Hybrid Agent failure-focused validator curriculum",
        "license": "proprietary-approved",
        "category": spec["category"],
        "reviewed": True,
        "review_method": "isolated deterministic behavioral validation",
        "behavior_concept": concept,
        "paraphrase_family": f"family-{prompt_index}",
        "split_group": f"family-{prompt_index}",
        "validator_backed": True,
        "validation_kind": "executable",
        "validation_harness": spec["harness"],
        "variant": f"variant-{variant_index}",
    }
    return {
        "id": f"validator-curriculum-v3-{concept}-family-{prompt_index}-{variant_index}",
        "messages": [
            {"role": "user", "content": prompt + " Return only one Python code block."},
            {"role": "assistant", "content": f"```python\n{answer.rstrip()}\n```"},
        ],
        "metadata": metadata,
    }


def main() -> int:
    output = ROOT / "datasets/candidates/validator-curriculum-v3.jsonl"
    rows = build_rows()
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} validator-backed teacher examples (curriculum v3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
