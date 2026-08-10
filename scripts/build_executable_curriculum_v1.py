#!/usr/bin/env python3
"""Build non-overlapping, validator-backed executable training families."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_agent.executable_evaluation import run_python_checks  # noqa: E402

SPECS = {
    "byte_commit": {
        "category": "coding", "harness": "commit_bytes_train_v1",
        "checks": ["writes_payload", "overwrites_old_payload", "rejects_non_bytes_without_damage", "leaves_no_staging_file"],
        "prompts": [
            "Implement commit_bytes(destination, payload) for durable replacement of a binary cache entry.",
            "Write commit_bytes(destination, payload) so an existing binary blob is never partially overwritten.",
            "Provide a standard-library commit_bytes(destination, payload) with failure cleanup.",
            "Create commit_bytes(destination, payload) using a same-directory staging file and atomic commit.",
            "A package publisher stores immutable binary wheels while concurrent readers poll the destination. Implement crash-conscious commit_bytes(destination, payload) so readers observe either the prior artifact or the complete new artifact.",
            "Write failure-safe code for commit_bytes(destination, payload), accepting bytes only.",
            "An image service refreshes thumbnail cache entries and must retain the previous bytes when encoding returns an invalid object. Define commit_bytes(destination, payload) with strict bytes validation and recoverable staging cleanup.",
            "Return Python implementing commit_bytes(destination, payload) without staging-file leaks.",
        ],
        "answers": [
            """import os\nimport tempfile\nfrom pathlib import Path\n\ndef commit_bytes(destination, payload):\n    if not isinstance(payload, bytes):\n        raise TypeError('payload must be bytes')\n    destination = Path(destination)\n    fd, staging = tempfile.mkstemp(dir=destination.parent, prefix=f'.{destination.name}.')\n    try:\n        with os.fdopen(fd, 'wb') as stream:\n            stream.write(payload)\n            stream.flush()\n            os.fsync(stream.fileno())\n        os.replace(staging, destination)\n    except BaseException:\n        try:\n            os.unlink(staging)\n        except FileNotFoundError:\n            pass\n        raise\n""",
            """import os\nfrom pathlib import Path\nfrom tempfile import NamedTemporaryFile\n\ndef commit_bytes(destination, payload):\n    if type(payload) is not bytes:\n        raise TypeError('bytes required')\n    destination = Path(destination)\n    temporary = None\n    try:\n        with NamedTemporaryFile('wb', dir=destination.parent, prefix='staging-', delete=False) as stream:\n            temporary = stream.name\n            stream.write(payload)\n            stream.flush()\n            os.fsync(stream.fileno())\n        os.replace(temporary, destination)\n        temporary = None\n    finally:\n        if temporary is not None:\n            try:\n                os.unlink(temporary)\n            except FileNotFoundError:\n                pass\n""",
        ],
    },
    "archive_containment": {
        "category": "coding", "harness": "archive_target_train_v1",
        "checks": ["maps_nested_member", "blocks_dotdot_member", "blocks_absolute_member", "blocks_linked_directory"],
        "prompts": [
            "Implement archive_target(extract_root, member_name) for safe archive extraction.",
            "Write archive_target(extract_root, member_name) that rejects Zip Slip and symlink escapes.",
            "Map an archive member below extract_root with archive_target(extract_root, member_name).",
            "Provide a containment-safe archive_target(extract_root, member_name).",
            "Implement archive_target for untrusted archive member names under one extraction directory.",
            "Define archive_target(extract_root, member_name), rejecting absolute and parent paths.",
            "Write the path gate archive_target(extract_root, member_name) for an unpacker.",
            "Return Python for archive_target that cannot traverse outside the extraction root.",
        ],
        "answers": [
            """from pathlib import Path\n\ndef archive_target(extract_root, member_name):\n    root = Path(extract_root).resolve()\n    member = Path(member_name)\n    if member.is_absolute():\n        raise ValueError('absolute archive member')\n    candidate = (root / member).resolve()\n    if not candidate.is_relative_to(root):\n        raise ValueError('archive member escapes root')\n    return candidate\n""",
            """from pathlib import Path\n\ndef archive_target(extract_root, member_name):\n    base = Path(extract_root).resolve(strict=True)\n    relative = Path(member_name)\n    if relative.is_absolute() or '..' in relative.parts:\n        raise PermissionError('unsafe archive member')\n    target = (base / relative).resolve()\n    try:\n        target.relative_to(base)\n    except ValueError as exc:\n        raise PermissionError('unsafe archive member') from exc\n    return target\n""",
        ],
    },
    "migration_state": {
        "category": "infrastructure", "harness": "migration_guard_train_v1",
        "checks": ["commits_verified_change", "failed_check_has_no_side_effect", "failed_verify_aborts"],
        "prompts": [
            "Implement apply_migration(plan, check, begin, verify, commit, abort) as a guarded state machine.",
            "Write apply_migration so prechecks precede changes and failed verification aborts.",
            "Define apply_migration(plan, check, begin, verify, commit, abort) with safe ordering.",
            "Implement a validation-gated database migration coordinator named apply_migration.",
            "Write apply_migration that commits only a verified migration and aborts a bad one.",
            "Provide failure-safe orchestration for apply_migration using the supplied callbacks.",
            "Implement apply_migration with no side effects after a failed precheck.",
            "Return Python for a check, begin, verify, commit-or-abort migration flow.",
        ],
        "answers": [
            """def apply_migration(plan, check, begin, verify, commit, abort):\n    if not check(plan):\n        raise RuntimeError('precheck failed')\n    begin(plan)\n    if not verify(plan):\n        abort(plan)\n        raise RuntimeError('verification failed')\n    commit(plan)\n""",
            """def apply_migration(plan, check, begin, verify, commit, abort):\n    approved = check(plan)\n    if approved is not True:\n        raise ValueError('migration rejected')\n    begin(plan)\n    verified = verify(plan)\n    if verified is True:\n        commit(plan)\n        return\n    abort(plan)\n    raise RuntimeError('migration verification failed')\n""",
        ],
    },
    "audit_trace": {
        "category": "tool_use", "harness": "audit_events_train_v1",
        "checks": ["records_request_and_outcome", "preserves_failure_status", "serializes_to_json"],
        "prompts": [
            "Implement audit_events(operation, input_data, status, detail) for a two-event audit record.",
            "Write audit_events returning JSON-ready request and outcome dictionaries.",
            "Define audit_events(operation, input_data, status, detail) without hiding failures.",
            "Create a serializable request/outcome pair with audit_events.",
            "Implement audit_events for honest operation status recording.",
            "Write audit_events that preserves structured outcome detail.",
            "Provide Python for audit_events returning ordered request and result records.",
            "Define the JSON-safe audit_events(operation, input_data, status, detail) helper.",
        ],
        "answers": [
            """def audit_events(operation, input_data, status, detail):\n    return [\n        {'kind': 'request', 'operation': operation, 'input': input_data},\n        {'kind': 'outcome', 'operation': operation, 'status': status, 'detail': detail},\n    ]\n""",
            """def audit_events(operation, input_data, status, detail):\n    request = {'kind': 'request', 'operation': operation, 'input': input_data}\n    outcome = {'kind': 'outcome', 'operation': operation, 'status': status, 'detail': detail}\n    return [request, outcome]\n""",
        ],
    },
}


def build_rows() -> list[dict]:
    rows = []
    for concept, spec in SPECS.items():
        for index, prompt in enumerate(spec["prompts"]):
            answer = spec["answers"][index % len(spec["answers"])]
            evaluator = {
                "checks": spec["checks"], "harness": spec["harness"],
                "language": "python", "type": "executable_assertions",
            }
            outcomes = run_python_checks(answer, evaluator)
            if not all(outcomes.values()):
                raise ValueError(f"Target failed validation: {concept}/family-{index}: {outcomes}")
            rows.append({
                "id": f"executable-curriculum-v1-{concept}-family-{index}",
                "messages": [
                    {"role": "user", "content": prompt + " Return only one Python code block."},
                    {"role": "assistant", "content": f"```python\n{answer.rstrip()}\n```"},
                ],
                "metadata": {
                    "source": "Hybrid Agent validator-backed executable curriculum",
                    "license": "proprietary-approved", "category": spec["category"],
                    "reviewed": True, "review_method": "isolated executable behavioral checks",
                    "behavior_concept": concept, "paraphrase_family": f"family-{index}",
                    "split_group": f"family-{index}", "executable_validated": True,
                    "validation_harness": spec["harness"],
                },
            })
    return sorted(rows, key=lambda row: row["id"])


def main() -> int:
    output = ROOT / "datasets/candidates/executable-curriculum-v1.jsonl"
    rows = build_rows()
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} validator-backed executable examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
