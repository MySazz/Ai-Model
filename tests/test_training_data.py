import json
from pathlib import Path

import pytest

from hybrid_agent.training_data import (
    LoadedRecord,
    load_jsonl,
    prepare_dataset,
    validate_records,
)


def valid_record(record_id: str = "coding-1") -> dict[str, object]:
    return {
        "id": record_id,
        "messages": [
            {"role": "user", "content": f"Question for {record_id}"},
            {"role": "assistant", "content": f"Verified answer for {record_id}"},
        ],
        "metadata": {
            "source": "human-authored test",
            "license": "proprietary-approved",
            "category": "coding",
            "reviewed": True,
            "human_reviewed": True,
            "review_method": "human-authored test review",
        },
    }


def loaded(record: dict[str, object], line: int = 1) -> LoadedRecord:
    return LoadedRecord(record=record, source="test.jsonl", line=line)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_valid_conversational_record_passes() -> None:
    report = validate_records([loaded(valid_record())])
    assert report.valid
    assert report.records == 1


def test_unapproved_license_and_review_are_errors() -> None:
    record = valid_record()
    record["metadata"]["license"] = "unknown"  # type: ignore[index]
    record["metadata"]["reviewed"] = False  # type: ignore[index]
    report = validate_records([loaded(record)])
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {"license", "reviewed"}


def test_secret_like_content_is_rejected() -> None:
    record = valid_record()
    record["messages"][0]["content"] = "api_key=super-secret-training-value"  # type: ignore[index]
    report = validate_records([loaded(record)])
    assert any(issue.code == "secret_detected" for issue in report.issues)
    assert not report.valid


def test_secret_in_tool_arguments_is_rejected() -> None:
    record = valid_record()
    record["messages"] = [
        {"role": "user", "content": "Call the service"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "request",
                        "arguments": {"api_key": "super-secret-tool-value"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "completed", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "The request completed."},
    ]
    report = validate_records([loaded(record)])
    assert any(issue.code == "secret_detected" for issue in report.issues)


def test_empty_assistant_content_is_allowed_for_tool_call() -> None:
    record = valid_record()
    record["messages"] = [
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "list_files", "arguments": {"path": "."}},
                }
            ],
        },
        {"role": "tool", "content": "README.md", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "README.md is present."},
    ]
    report = validate_records([loaded(record)])
    assert report.valid


def test_possible_pii_requires_warning_but_not_failure() -> None:
    record = valid_record()
    record["messages"][0]["content"] = "Email example@example.com"  # type: ignore[index]
    report = validate_records([loaded(record)])
    assert report.valid
    assert any(issue.code == "possible_pii" for issue in report.issues)


def test_duplicate_ids_and_content_are_rejected() -> None:
    first = valid_record("duplicate")
    second = json.loads(json.dumps(first))
    report = validate_records([loaded(first), loaded(second, line=2)])
    codes = {issue.code for issue in report.issues}
    assert "duplicate_id" in codes
    assert "exact_duplicate" in codes


def test_near_duplicate_is_rejected_to_prevent_split_leakage() -> None:
    first = valid_record("first")
    first["messages"][1]["content"] = (  # type: ignore[index]
        "Inspect the evidence, identify the cause, make a minimal change, "
        "run the relevant tests, review the output, and report verified results."
    )
    second = json.loads(json.dumps(first))
    second["id"] = "second"
    second["messages"][1]["content"] += " Additional."  # type: ignore[index]
    report = validate_records([loaded(first), loaded(second, line=2)])
    assert any(issue.code == "near_duplicate" for issue in report.issues)
    assert not report.valid


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")
    records, issues = load_jsonl([path])
    report = validate_records(records, issues)
    assert not report.valid
    assert report.issues[0].code == "invalid_json"


def test_malformed_tool_call_is_rejected() -> None:
    record = valid_record()
    record["messages"] = [
        {"role": "user", "content": "Use a tool"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {}}]},
    ]
    report = validate_records([loaded(record)])
    assert any(issue.code == "tool_call_schema" for issue in report.issues)
    assert not report.valid


def test_oversized_jsonl_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    path.write_text("{\"padding\":\"" + "x" * 2_000_001 + "\"}\n", encoding="utf-8")
    records, issues = load_jsonl([path])
    assert records == []
    assert issues[0].code == "line_too_large"


def test_prepare_is_deterministic_and_writes_hash_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [valid_record(f"record-{index}") for index in range(30)])
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_dataset([source], first_dir, seed="stable-seed")
    second = prepare_dataset([source], second_dir, seed="stable-seed")

    assert first["counts"] == second["counts"]
    assert first["output_sha256"] == second["output_sha256"]
    assert sum(first["counts"].values()) == 30
    for split in ("train", "validation", "test"):
        assert (first_dir / f"{split}.jsonl").read_bytes() == (
            second_dir / f"{split}.jsonl"
        ).read_bytes()
    assert (first_dir / "manifest.json").is_file()


def test_prepare_refuses_invalid_data(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    record = valid_record()
    record["metadata"]["reviewed"] = False  # type: ignore[index]
    write_jsonl(source, [record])
    with pytest.raises(ValueError, match="validation error"):
        prepare_dataset([source], tmp_path / "output")


def test_split_ratios_must_leave_test_partition(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [valid_record()])
    with pytest.raises(ValueError, match="positive test"):
        prepare_dataset(
            [source],
            tmp_path / "output",
            train_ratio=0.9,
            validation_ratio=0.1,
        )


def test_prepare_keeps_split_groups_together(tmp_path: Path) -> None:
    source = tmp_path / "grouped.jsonl"
    records = []
    for family in range(30):
        for variant in range(3):
            record = valid_record(f"family-{family}-variant-{variant}")
            record["metadata"]["split_group"] = f"family-{family}"  # type: ignore[index]
            records.append(record)
    write_jsonl(source, records)

    output = tmp_path / "output"
    manifest = prepare_dataset([source], output, seed="family-split-v1")

    locations: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        for row in load_jsonl([output / f"{split}.jsonl"])[0]:
            group = row.record["metadata"]["split_group"]
            locations.setdefault(group, set()).add(split)
    assert all(len(splits) == 1 for splits in locations.values())
    assert manifest["split_strategy"] == {
        "type": "metadata_group",
        "metadata_field": "split_group",
    }


def test_prepare_refuses_partially_grouped_dataset(tmp_path: Path) -> None:
    source = tmp_path / "partially-grouped.jsonl"
    first = valid_record("grouped")
    first["metadata"]["split_group"] = "family-a"  # type: ignore[index]
    write_jsonl(source, [first, valid_record("ungrouped")])

    with pytest.raises(ValueError, match="mixes grouped and ungrouped"):
        prepare_dataset([source], tmp_path / "output")


def test_invalid_split_group_is_rejected() -> None:
    record = valid_record()
    record["metadata"]["split_group"] = ""  # type: ignore[index]
    report = validate_records([loaded(record)])
    assert not report.valid
    assert any(issue.code == "split_group" for issue in report.issues)
