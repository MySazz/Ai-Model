"""Child-process harnesses for executable evaluation; invoked with Python -I."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def load(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load submission")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attempt(function: Callable[[], bool]) -> bool:
    try:
        return function() is True
    except BaseException:
        return False


def atomic_json(module: Any) -> dict[str, bool]:
    def writes_valid_json() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            module.save_json_atomic(path, {"value": [1, 2]})
            return json.loads(path.read_text()) == {"value": [1, 2]}

    def replaces_existing() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"old": true}')
            module.save_json_atomic(path, {"new": True})
            return json.loads(path.read_text()) == {"new": True}

    def preserves_on_failure() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"old": true}')
            try:
                module.save_json_atomic(path, {"bad": object()})
            except Exception:
                pass
            return json.loads(path.read_text()) == {"old": True}

    def cleans_temp_on_failure() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("{}")
            before = set(Path(directory).iterdir())
            try:
                module.save_json_atomic(path, {"bad": object()})
            except Exception:
                pass
            return set(Path(directory).iterdir()) == before

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def rooted_path(module: Any) -> dict[str, bool]:
    def accepts_child() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "nested" / "file.txt"
            return Path(module.safe_join(root, "nested/file.txt")) == expected

    def rejects_parent_escape() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.safe_join(Path(directory), "../outside.txt")
            except (ValueError, PermissionError):
                return True
            return False

    def rejects_absolute_escape() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.safe_join(Path(directory), "/tmp/outside.txt")
            except (ValueError, PermissionError):
                return True
            return False

    def rejects_symlink_escape() -> bool:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "link").symlink_to(outside, target_is_directory=True)
            try:
                module.safe_join(root, "link/secret.txt")
            except (ValueError, PermissionError):
                return True
            return False

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def controlled_rollout(module: Any) -> dict[str, bool]:
    def success_order() -> bool:
        events: list[str] = []
        module.controlled_rollout(
            "r1", lambda _r: events.append("validate") or True,
            lambda _r, pct: events.append(f"shift:{pct}"),
            lambda _r: events.append("monitor") or True,
            lambda _r: events.append("rollback"),
        )
        return events == ["validate", "shift:10", "monitor", "shift:100"]

    def validation_stops_rollout() -> bool:
        events: list[str] = []
        try:
            module.controlled_rollout(
                "r1", lambda _r: events.append("validate") or False,
                lambda _r, pct: events.append(f"shift:{pct}"),
                lambda _r: events.append("monitor") or True,
                lambda _r: events.append("rollback"),
            )
        except Exception:
            pass
        return events == ["validate"]

    def rollback_on_bad_monitor() -> bool:
        events: list[str] = []
        try:
            module.controlled_rollout(
                "r1", lambda _r: events.append("validate") or True,
                lambda _r, pct: events.append(f"shift:{pct}"),
                lambda _r: events.append("monitor") or False,
                lambda _r: events.append("rollback"),
            )
        except Exception:
            pass
        return events == ["validate", "shift:10", "monitor", "rollback"]

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def structured_trace(module: Any) -> dict[str, bool]:
    def valid_success_trace() -> bool:
        trace = module.build_tool_trace("read_file", {"path": "a.txt"}, True, "ok")
        return (
            isinstance(trace, list) and len(trace) == 2
            and trace[0] == {"event": "tool_call", "tool": "read_file", "arguments": {"path": "a.txt"}}
            and trace[1] == {"event": "tool_result", "tool": "read_file", "ok": True, "output": "ok"}
        )

    def valid_failure_trace() -> bool:
        trace = module.build_tool_trace("read_file", {}, False, "denied")
        return (
            isinstance(trace, list) and len(trace) == 2
            and trace[1].get("ok") is False and trace[1].get("output") == "denied"
            and trace[1].get("tool") == "read_file"
        )

    def json_serializable() -> bool:
        json.dumps(module.build_tool_trace("search", {"q": "x"}, True, ["a"]))
        return True

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def commit_bytes(module: Any) -> dict[str, bool]:
    def writes_payload() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            module.commit_bytes(path, b"new\x00data")
            return path.read_bytes() == b"new\x00data"

    def overwrites_old_payload() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"old")
            module.commit_bytes(path, b"new")
            return path.read_bytes() == b"new"

    def rejects_non_bytes_without_damage() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"old")
            try:
                module.commit_bytes(path, object())
            except (TypeError, ValueError):
                pass
            return path.read_bytes() == b"old"

    def leaves_no_staging_file() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            before = set(Path(directory).iterdir())
            try:
                module.commit_bytes(path, object())
            except (TypeError, ValueError):
                pass
            return set(Path(directory).iterdir()) == before

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def archive_target(module: Any) -> dict[str, bool]:
    def maps_nested_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            return Path(module.archive_target(root, "images/icon.png")) == root / "images/icon.png"

    def blocks_dotdot_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.archive_target(Path(directory), "../../escape")
            except (ValueError, PermissionError):
                return True
            return False

    def blocks_absolute_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.archive_target(Path(directory), "/etc/passwd")
            except (ValueError, PermissionError):
                return True
            return False

    def blocks_linked_directory() -> bool:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "linked").symlink_to(outside, target_is_directory=True)
            try:
                module.archive_target(root, "linked/file")
            except (ValueError, PermissionError):
                return True
            return False

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def migration_guard(module: Any) -> dict[str, bool]:
    def commits_verified_change() -> bool:
        events: list[str] = []
        module.apply_migration(
            "m1", lambda _m: events.append("check") or True,
            lambda _m: events.append("begin"), lambda _m: events.append("verify") or True,
            lambda _m: events.append("commit"), lambda _m: events.append("abort"),
        )
        return events == ["check", "begin", "verify", "commit"]

    def failed_check_has_no_side_effect() -> bool:
        events: list[str] = []
        try:
            module.apply_migration(
                "m1", lambda _m: events.append("check") or False,
                lambda _m: events.append("begin"), lambda _m: events.append("verify") or True,
                lambda _m: events.append("commit"), lambda _m: events.append("abort"),
            )
        except Exception:
            pass
        return events == ["check"]

    def failed_verify_aborts() -> bool:
        events: list[str] = []
        try:
            module.apply_migration(
                "m1", lambda _m: events.append("check") or True,
                lambda _m: events.append("begin"), lambda _m: events.append("verify") or False,
                lambda _m: events.append("commit"), lambda _m: events.append("abort"),
            )
        except Exception:
            pass
        return events == ["check", "begin", "verify", "abort"]

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def audit_events(module: Any) -> dict[str, bool]:
    def records_request_and_outcome() -> bool:
        events = module.audit_events("delete", {"id": 7}, "denied", "policy")
        return events == [
            {"kind": "request", "operation": "delete", "input": {"id": 7}},
            {"kind": "outcome", "operation": "delete", "status": "denied", "detail": "policy"},
        ]

    def preserves_failure_status() -> bool:
        events = module.audit_events("fetch", {}, "error", {"code": 500})
        return events[1]["status"] == "error" and events[1]["detail"] == {"code": 500}

    def serializes_to_json() -> bool:
        json.dumps(module.audit_events("read", {"x": 1}, "ok", [1, 2]))
        return True

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


HARNESSES = {
    "atomic_json_v1": atomic_json,
    "rooted_path_v1": rooted_path,
    "controlled_rollout_v1": controlled_rollout,
    "structured_trace_v1": structured_trace,
    "commit_bytes_train_v1": commit_bytes,
    "archive_target_train_v1": archive_target,
    "migration_guard_train_v1": migration_guard,
    "audit_events_train_v1": audit_events,
}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in HARNESSES:
        raise SystemExit(2)
    module = load(sys.argv[2])
    outcomes = HARNESSES[sys.argv[1]](module)
    sys.stdout.write(json.dumps(outcomes, sort_keys=True))


if __name__ == "__main__":
    main()
