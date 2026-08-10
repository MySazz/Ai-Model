"""Dataset validation and deterministic preparation for supervised fine-tuning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .privacy import PrivacyClassifier, PrivacyLevel


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
ALLOWED_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MIT",
    "proprietary-approved",
    "public-domain",
}
ALLOWED_CATEGORIES = {
    "coding", "tool_use", "research", "vision", "infrastructure", "safety", "general"
}
NEAR_DUPLICATE_THRESHOLD = 0.8
MAX_JSONL_LINE_CHARS = 2_000_000
MAX_RECORDS = 10_000
MAX_MESSAGES_PER_RECORD = 100
MAX_MESSAGE_CHARS = 200_000
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    source: str
    line: int
    record_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    records: int
    errors: int
    warnings: int
    issues: tuple[DatasetIssue, ...]

    @property
    def valid(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "errors": self.errors,
            "warnings": self.warnings,
            "valid": self.valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class LoadedRecord:
    record: dict[str, Any]
    source: str
    line: int


def load_jsonl(paths: Iterable[Path]) -> tuple[list[LoadedRecord], list[DatasetIssue]]:
    records: list[LoadedRecord] = []
    issues: list[DatasetIssue] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as stream:
                line_number = 0
                while True:
                    raw_line = stream.readline(MAX_JSONL_LINE_CHARS + 1)
                    if not raw_line:
                        break
                    line_number += 1
                    oversized = len(raw_line) > MAX_JSONL_LINE_CHARS
                    while oversized and raw_line and not raw_line.endswith("\n"):
                        raw_line = stream.readline(MAX_JSONL_LINE_CHARS + 1)
                    if oversized:
                        issues.append(
                            DatasetIssue(
                                "error",
                                "line_too_large",
                                f"JSONL line exceeds {MAX_JSONL_LINE_CHARS} characters.",
                                str(path),
                                line_number,
                            )
                        )
                        continue
                    if not raw_line.strip():
                        continue
                    try:
                        value = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        issues.append(
                            DatasetIssue(
                                "error",
                                "invalid_json",
                                exc.msg,
                                str(path),
                                line_number,
                            )
                        )
                        continue
                    if not isinstance(value, dict):
                        issues.append(
                            DatasetIssue(
                                "error",
                                "record_type",
                                "Each JSONL line must contain an object.",
                                str(path),
                                line_number,
                            )
                        )
                        continue
                    records.append(LoadedRecord(value, str(path), line_number))
                    if len(records) > MAX_RECORDS:
                        issues.append(
                            DatasetIssue(
                                "error",
                                "record_limit",
                                f"Dataset exceeds the {MAX_RECORDS}-record validation limit.",
                                str(path),
                                line_number,
                            )
                        )
                        return records, issues
        except (OSError, UnicodeError) as exc:
            issues.append(
                DatasetIssue("error", "file_read", str(exc), str(path), 0)
            )
            continue
    return records, issues


def validate_records(
    loaded: Iterable[LoadedRecord],
    initial_issues: Iterable[DatasetIssue] = (),
) -> ValidationReport:
    issues = list(initial_issues)
    records = list(loaded)
    seen_ids: dict[str, LoadedRecord] = {}
    seen_content: dict[str, LoadedRecord] = {}
    normalized_for_similarity: list[tuple[LoadedRecord, set[str]]] = []

    for item in records:
        unknown_fields = sorted(set(item.record) - {"id", "messages", "metadata", "tools"})
        if unknown_fields:
            _issue(
                issues,
                item,
                "error",
                "record_fields",
                "Unsupported top-level fields: " + ", ".join(unknown_fields),
            )
        record_id = item.record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            _issue(issues, item, "error", "missing_id", "Record requires a non-empty string id.")
            record_id = None
        elif record_id in seen_ids:
            _issue(issues, item, "error", "duplicate_id", f"Duplicate record id: {record_id}")
        else:
            seen_ids[record_id] = item

        messages = item.record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            _issue(
                issues,
                item,
                "error",
                "messages",
                "Record requires at least two messages.",
                record_id,
            )
            continue
        if len(messages) > MAX_MESSAGES_PER_RECORD:
            _issue(
                issues,
                item,
                "error",
                "message_limit",
                f"Record exceeds {MAX_MESSAGES_PER_RECORD} messages.",
                record_id,
            )
        assistant_count = 0
        previous_role: str | None = None
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                _issue(
                    issues,
                    item,
                    "error",
                    "message_type",
                    f"Message {index} must be an object.",
                    record_id,
                )
                continue
            role = message.get("role")
            content = message.get("content")
            unknown_message_fields = sorted(
                set(message) - {"role", "content", "tool_calls", "name", "tool_call_id"}
            )
            if unknown_message_fields:
                _issue(
                    issues,
                    item,
                    "error",
                    "message_fields",
                    f"Message {index} has unsupported fields: "
                    + ", ".join(unknown_message_fields),
                    record_id,
                )
            if role not in ALLOWED_ROLES:
                _issue(
                    issues,
                    item,
                    "error",
                    "message_role",
                    f"Message {index} has unsupported role: {role!r}.",
                    record_id,
                )
            has_tool_calls = (
                role == "assistant"
                and isinstance(message.get("tool_calls"), list)
                and bool(message["tool_calls"])
            )
            if not isinstance(content, str) or (not content.strip() and not has_tool_calls):
                _issue(
                    issues,
                    item,
                    "error",
                    "message_content",
                    f"Message {index} requires non-empty text content.",
                    record_id,
                )
            elif len(content) > MAX_MESSAGE_CHARS:
                _issue(
                    issues,
                    item,
                    "error",
                    "message_too_large",
                    f"Message {index} exceeds {MAX_MESSAGE_CHARS} characters.",
                    record_id,
                )
            if has_tool_calls:
                _validate_tool_calls(item, issues, record_id, index, message["tool_calls"])
            if role == "assistant":
                assistant_count += 1
            if role == "system" and index != 0:
                _issue(
                    issues,
                    item,
                    "warning",
                    "system_position",
                    "System messages should normally appear first.",
                    record_id,
                )
            if role == previous_role and role in {"user", "assistant"}:
                _issue(
                    issues,
                    item,
                    "warning",
                    "repeated_role",
                    f"Consecutive {role} messages may indicate malformed conversation data.",
                    record_id,
                )
            previous_role = role if isinstance(role, str) else previous_role

        if assistant_count == 0:
            _issue(
                issues,
                item,
                "error",
                "missing_assistant",
                "SFT records require at least one assistant response.",
                record_id,
            )

        _validate_metadata(item, issues, record_id)
        full_text = "\n".join(
            _flatten_text(
                {
                    "messages": messages,
                    "tools": item.record.get("tools", []),
                }
            )
        )
        privacy = PrivacyClassifier().classify(full_text)
        if privacy.level is PrivacyLevel.LOCAL_ONLY:
            _issue(
                issues,
                item,
                "error",
                "secret_detected",
                "Secret-like content detected: " + ", ".join(privacy.reasons),
                record_id,
            )
        if EMAIL_PATTERN.search(full_text) or PHONE_PATTERN.search(full_text):
            _issue(
                issues,
                item,
                "warning",
                "possible_pii",
                "Possible email address or phone number requires human review.",
                record_id,
            )

        fingerprint = _content_fingerprint(messages)
        if fingerprint in seen_content:
            other = seen_content[fingerprint].record.get("id", "unknown")
            _issue(
                issues,
                item,
                "error",
                "exact_duplicate",
                f"Conversation duplicates record {other}.",
                record_id,
            )
        else:
            seen_content[fingerprint] = item
        normalized_for_similarity.append((item, _token_shingles(full_text)))

    _detect_near_duplicates(normalized_for_similarity, issues)
    return ValidationReport(
        records=len(records),
        errors=sum(issue.severity == "error" for issue in issues),
        warnings=sum(issue.severity == "warning" for issue in issues),
        issues=tuple(issues),
    )


def prepare_dataset(
    input_paths: list[Path],
    output_dir: Path,
    *,
    seed: str = "hybrid-agent-v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, Any]:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("Split ratios must be between zero and one.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave a positive test split.")

    loaded, load_issues = load_jsonl(input_paths)
    report = validate_records(loaded, load_issues)
    if not report.valid:
        raise ValueError(f"Dataset has {report.errors} validation error(s).")

    split_groups = [_record_split_group(item.record) for item in loaded]
    grouped_split = any(group is not None for group in split_groups)
    if grouped_split and any(group is None for group in split_groups):
        raise ValueError(
            "Dataset mixes grouped and ungrouped records; every record must define "
            "metadata.split_group when grouped splitting is used."
        )

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for item in loaded:
        normalized = normalize_record(item.record)
        split_key = _record_split_group(normalized) or str(normalized["id"])
        bucket = _split_bucket(split_key, seed)
        if bucket < train_ratio:
            split = "train"
        elif bucket < train_ratio + validation_ratio:
            split = "validation"
        else:
            split = "test"
        splits[split].append(normalized)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for split, records in splits.items():
        records.sort(key=lambda record: str(record["id"]))
        output_hashes[split] = _atomic_write_jsonl(
            output_dir / f"{split}.jsonl", records
        )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": round(1 - train_ratio - validation_ratio, 12),
        },
        "split_strategy": {
            "type": "metadata_group" if grouped_split else "record_id",
            "metadata_field": "split_group" if grouped_split else None,
        },
        "input_files": [
            {
                "path": _portable_path(path),
                "sha256": _sha256_file(path),
            }
            for path in input_paths
        ],
        "counts": {split: len(records) for split, records in splits.items()},
        "output_sha256": output_hashes,
        "validation": report.to_dict(),
    }
    _atomic_write_text(
        output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "id": str(record["id"]).strip(),
        "messages": [
            {
                key: message[key]
                for key in ("role", "content", "tool_calls", "name", "tool_call_id")
                if key in message
            }
            for message in record["messages"]
        ],
        "metadata": {
            key: record["metadata"][key]
            for key in sorted(record["metadata"])
        },
    }
    if "tools" in record:
        normalized["tools"] = record["tools"]
    return normalized


def _validate_metadata(
    item: LoadedRecord,
    issues: list[DatasetIssue],
    record_id: str | None,
) -> None:
    metadata = item.record.get("metadata")
    if not isinstance(metadata, dict):
        _issue(
            issues,
            item,
            "error",
            "metadata",
            "Record requires a metadata object.",
            record_id,
        )
        return
    required = {"source", "license", "category", "reviewed"}
    missing = sorted(required - metadata.keys())
    if missing:
        _issue(
            issues,
            item,
            "error",
            "metadata_fields",
            "Missing metadata fields: " + ", ".join(missing),
            record_id,
        )
    if not isinstance(metadata.get("source"), str) or not metadata.get("source", "").strip():
        _issue(
            issues,
            item,
            "error",
            "source",
            "metadata.source must be a non-empty provenance description.",
            record_id,
        )
    license_name = metadata.get("license")
    if license_name not in ALLOWED_LICENSES:
        _issue(
            issues,
            item,
            "error",
            "license",
            f"License is missing or not approved: {license_name!r}.",
            record_id,
        )
    category = metadata.get("category")
    if category not in ALLOWED_CATEGORIES:
        _issue(
            issues,
            item,
            "error",
            "category",
            f"Unsupported capability category: {category!r}.",
            record_id,
        )
    if metadata.get("reviewed") is not True:
        _issue(
            issues,
            item,
            "error",
            "reviewed",
            "metadata.reviewed must be true after the declared quality review completes.",
            record_id,
        )
    review_method = metadata.get("review_method")
    if metadata.get("reviewed") is True and (
        not isinstance(review_method, str) or not review_method.strip()
    ):
        _issue(
            issues,
            item,
            "warning",
            "review_provenance",
            "Reviewed records should declare metadata.review_method.",
            record_id,
        )
    if "human_reviewed" in metadata and not isinstance(metadata["human_reviewed"], bool):
        _issue(
            issues,
            item,
            "error",
            "human_reviewed",
            "metadata.human_reviewed must be boolean when provided.",
            record_id,
        )
    if metadata.get("calibration_only") is True:
        _issue(
            issues,
            item,
            "error",
            "calibration_only",
            "Calibration-only records cannot enter prepared training datasets.",
            record_id,
        )
    if "split_group" in metadata and (
        not isinstance(metadata["split_group"], str)
        or not metadata["split_group"].strip()
    ):
        _issue(
            issues,
            item,
            "error",
            "split_group",
            "metadata.split_group must be a non-empty string when provided.",
            record_id,
        )


def _issue(
    issues: list[DatasetIssue],
    item: LoadedRecord,
    severity: str,
    code: str,
    message: str,
    record_id: str | None = None,
) -> None:
    issues.append(
        DatasetIssue(
            severity=severity,
            code=code,
            message=message,
            source=item.source,
            line=item.line,
            record_id=record_id,
        )
    )


def _content_fingerprint(messages: list[Any]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _flatten_text(value: Any, key: str = "") -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for child_key in sorted(value):
            flattened.extend(_flatten_text(value[child_key], str(child_key)))
        return flattened
    if isinstance(value, list):
        flattened = []
        for child in value:
            flattened.extend(_flatten_text(child, key))
        return flattened
    if value is None:
        return []
    rendered = str(value)
    return [f"{key}={rendered}" if key else rendered]


def _token_shingles(text: str, size: int = 3) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _detect_near_duplicates(
    records: list[tuple[LoadedRecord, set[str]]],
    issues: list[DatasetIssue],
) -> None:
    if len(records) > 10000:
        first = records[0][0]
        _issue(
            issues,
            first,
            "error",
            "near_duplicate_limit",
            "Near-duplicate validation refuses datasets above 10,000 records; shard and "
            "deduplicate before preparation.",
            str(first.record.get("id")) if first.record.get("id") else None,
        )
        return
    for index, (left, left_shingles) in enumerate(records):
        if not left_shingles:
            continue
        for right, right_shingles in records[index + 1 :]:
            union = left_shingles | right_shingles
            similarity = len(left_shingles & right_shingles) / len(union) if union else 0
            if similarity >= NEAR_DUPLICATE_THRESHOLD and _content_fingerprint(
                left.record.get("messages", [])
            ) != _content_fingerprint(right.record.get("messages", [])):
                _issue(
                    issues,
                    right,
                    "error",
                    "near_duplicate",
                    (
                        f"Conversation is {similarity:.0%} similar to "
                        f"record {left.record.get('id', 'unknown')}."
                    ),
                    str(right.record.get("id")) if right.record.get("id") else None,
                )


def _split_bucket(record_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _record_split_group(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    group = metadata.get("split_group")
    return group.strip() if isinstance(group, str) and group.strip() else None


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    current = Path.cwd().resolve()
    if resolved == current or current in resolved.parents:
        return str(resolved.relative_to(current))
    return resolved.name


def _validate_tool_calls(
    item: LoadedRecord,
    issues: list[DatasetIssue],
    record_id: str | None,
    message_index: int,
    calls: list[Any],
) -> None:
    for call_index, call in enumerate(calls):
        valid = False
        if isinstance(call, dict) and call.get("type", "function") == "function":
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                valid = (
                    isinstance(name, str)
                    and bool(name)
                    and isinstance(function.get("arguments", {}), dict)
                )
        if not valid:
            _issue(
                issues,
                item,
                "error",
                "tool_call_schema",
                f"Message {message_index} tool call {call_index} is malformed.",
                record_id,
            )


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                line = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream.write(line)
                digest.update(line.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest()
