#!/usr/bin/env python3
"""Build the pinned capability-v1 SFT corpus from approved upstream sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCES = {
    "code": {
        "url": "https://huggingface.co/datasets/flwrlabs/code-alpaca-20k/resolve/993cf40c2e806a2641f388fb4026f30c4c73dc97/code_alpaca_20k.json",
        "filename": "hybrid-code-alpaca.json",
        "revision": "993cf40c2e806a2641f388fb4026f30c4c73dc97",
        "license": "Apache-2.0",
        "target": 1500,
    },
    "tool": {
        "url": "https://huggingface.co/datasets/Johin/function-calling-dataset/resolve/ef3f5c4ce7cbf80b55f017fdb8695226cfad0976/data/train.jsonl",
        "filename": "hybrid-function-calling.jsonl",
        "revision": "ef3f5c4ce7cbf80b55f017fdb8695226cfad0976",
        "license": "Apache-2.0",
        "target": 1000,
    },
    "infrastructure": {
        "url": "https://huggingface.co/datasets/jalpan04/devops-sft-dataset/resolve/bd57569ae3424607bf8a404e274429f2182f30ea/train.jsonl",
        "filename": "hybrid-devops.jsonl",
        "revision": "bd57569ae3424607bf8a404e274429f2182f30ea",
        "license": "Apache-2.0",
        "target": 700,
    },
}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
SECRET = re.compile(r"(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]{8,}", re.I)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/raw/cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/candidates/capability-v1.jsonl")
    )
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/capability-v1.sources.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.source_dir.mkdir(parents=True, exist_ok=True)

    source_paths: dict[str, Path] = {}
    for name, source in SOURCES.items():
        path = args.source_dir / str(source["filename"])
        if not path.is_file():
            if not args.download:
                raise SystemExit(f"Missing {path}; rerun with --download.")
            with urllib.request.urlopen(str(source["url"]), timeout=120) as response:
                path.write_bytes(response.read())
        source_paths[name] = path

    evaluation_prompts = load_evaluation_prompts(Path("datasets/evaluations"))
    records: list[dict[str, Any]] = []
    records.extend(select("code", code_records(source_paths["code"]), evaluation_prompts))
    records.extend(select("tool", tool_records(source_paths["tool"]), evaluation_prompts))
    records.extend(
        select("infrastructure", devops_records(source_paths["infrastructure"]), evaluation_prompts)
    )
    records.sort(key=lambda record: record["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "output": str(args.output),
        "output_sha256": digest(args.output),
        "records": len(records),
        "counts": {name: int(source["target"]) for name, source in SOURCES.items()},
        "selection": "ascending SHA-256 of normalized source record after local quality filters",
        "review_method": "upstream-curated plus deterministic local format, privacy, length, duplicate, and evaluation-overlap filters",
        "sources": [
            {
                "name": name,
                "url": source["url"],
                "revision": source["revision"],
                "license": source["license"],
                "download_sha256": digest(source_paths[name]),
            }
            for name, source in SOURCES.items()
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


def code_records(path: Path) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        prompt = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "")).strip()
        if extra:
            prompt += "\n\nInput:\n" + extra
        yield make_record("code", index, prompt, str(row.get("output", "")), "coding")


def tool_records(path: Path) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(read_jsonl(path)):
        messages = []
        last_call_id: str | None = None
        for offset, message in enumerate(row.get("messages", [])):
            role = message.get("role")
            content = message.get("content")
            if role == "function":
                role = "tool"
            converted: dict[str, Any] = {"role": role, "content": content or ""}
            call = message.get("function_call")
            if role == "assistant" and isinstance(call, dict):
                try:
                    arguments = json.loads(call.get("arguments", "{}"))
                except (TypeError, json.JSONDecodeError):
                    arguments = call.get("arguments", "{}")
                last_call_id = f"call-{index}-{offset}"
                converted["tool_calls"] = [
                    {
                        "id": last_call_id,
                        "type": "function",
                        "function": {"name": call.get("name"), "arguments": arguments},
                    }
                ]
            elif role == "tool" and last_call_id:
                converted["tool_call_id"] = last_call_id
            messages.append(converted)
        yield make_record("tool", index, "", "", "general", messages=messages)


def devops_records(path: Path) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(read_jsonl(path)):
        messages = row.get("messages", [])
        yield make_record("infra", index, "", "", "infrastructure", messages=messages)


def make_record(
    prefix: str,
    index: int,
    prompt: str,
    answer: str,
    category: str,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"capability-v1-{prefix}-{index:06d}",
        "messages": messages
        if messages is not None
        else [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        "metadata": {
            "source": f"pinned upstream {prefix} source; deterministic local selection",
            "license": "Apache-2.0",
            "category": category,
            "reviewed": True,
            "review_method": "upstream-curated-and-automated-filtered",
        },
    }


def select(
    source_name: str, candidates: Iterable[dict[str, Any]], evaluation_prompts: list[set[str]]
) -> list[dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    for record in candidates:
        messages = record["messages"]
        if not valid_messages(messages):
            continue
        text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if not 40 <= len(text) <= 12000 or EMAIL.search(text) or PHONE.search(text) or SECRET.search(text):
            continue
        fingerprint = hashlib.sha256(normalize(text).encode()).hexdigest()
        if fingerprint in accepted:
            continue
        prompt_tokens = tokens(str(record["messages"][-2].get("content", ""))) if len(record["messages"]) >= 2 else set()
        if any(jaccard(prompt_tokens, held_out) >= 0.45 for held_out in evaluation_prompts):
            continue
        accepted[fingerprint] = record
    target = int(SOURCES[source_name]["target"])
    selected: list[dict[str, Any]] = []
    selected_shingles: list[set[str]] = []
    for key in sorted(accepted):
        record = accepted[key]
        candidate_shingles = shingles(json.dumps(record["messages"], sort_keys=True))
        if any(jaccard(candidate_shingles, prior) >= 0.8 for prior in selected_shingles):
            continue
        selected.append(record)
        selected_shingles.append(candidate_shingles)
        if len(selected) == target:
            return selected
    raise SystemExit(f"Only {len(selected)} diverse {source_name} records; need {target}.")


def valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    assistant = False
    previous = None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            return False
        role = message["role"]
        if role == "assistant":
            assistant = True
        if role in {"user", "assistant"} and role == previous:
            return False
        content = message.get("content")
        if (not isinstance(content, str) or not content.strip()) and not (
            role == "assistant" and message.get("tool_calls")
        ):
            return False
        previous = role
    return assistant


def load_evaluation_prompts(root: Path) -> list[set[str]]:
    prompts = []
    for path in sorted(root.glob("*.jsonl")):
        for row in read_jsonl(path):
            if isinstance(row.get("prompt"), str):
                prompts.append(tokens(row["prompt"]))
    return prompts


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.casefold()))


def shingles(text: str, size: int = 3) -> set[str]:
    ordered = re.findall(r"[a-z0-9_]+", text.casefold())
    if len(ordered) < size:
        return {" ".join(ordered)} if ordered else set()
    return {" ".join(ordered[index : index + size]) for index in range(len(ordered) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.casefold()))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
