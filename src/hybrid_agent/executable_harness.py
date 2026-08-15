"""Child-process harnesses for executable evaluation; invoked with Python -I."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def install_safety_audit_hook(submission_path: Path) -> None:
    """Deny network/process access and confine file writes during authored checks.

    This is defense in depth for accidental or ordinary Python behavior. The parent
    process still requires a disposable VM or container as the hostile-code boundary.
    """
    evaluation_root = submission_path.parent.resolve()
    scratch = evaluation_root / "scratch"
    scratch.mkdir(mode=0o700)
    tempfile.tempdir = str(scratch)
    read_roots = tuple(
        dict.fromkeys(
            path.resolve()
            for path in (
                evaluation_root,
                Path(__file__).parent,
                Path(sys.base_prefix),
                Path(sys.prefix),
            )
        )
    )
    write_roots = (evaluation_root,)
    blocked_events = {
        "ctypes.dlopen",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "subprocess.Popen",
    }

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in blocked_events or event.startswith("socket."):
            raise PermissionError(f"Executable evaluation blocked operation: {event}")
        if event == "open" and args and isinstance(args[0], str | bytes | os.PathLike):
            path = Path(os.fsdecode(args[0]))
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writing = (
                isinstance(mode, str) and any(character in mode for character in "wax+")
            ) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
            allowed = _inside(path, write_roots if writing else read_roots)
            if not allowed:
                raise PermissionError(f"Executable evaluation blocked file access: {path}")
        if event in {
            "os.chdir",
            "os.chmod",
            "os.chown",
            "os.link",
            "os.mkdir",
            "os.remove",
            "os.rename",
            "os.rmdir",
            "os.symlink",
            "os.truncate",
            "os.utime",
        }:
            paths = [
                arg for arg in args[:2] if isinstance(arg, str | bytes | os.PathLike)
            ]
            if any(not _inside(Path(os.fsdecode(path)), write_roots) for path in paths):
                raise PermissionError(f"Executable evaluation blocked operation: {event}")

    sys.addaudithook(audit)


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


def commit_bytes_v2(module: Any) -> dict[str, bool]:
    """Exercise strict payload handling and cleanup on real commit failures."""

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

    def writes_empty_payload() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"old")
            module.commit_bytes(path, b"")
            return path.read_bytes() == b""

    def rejects_non_bytes_without_damage() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"old")
            try:
                module.commit_bytes(path, object())
            except (TypeError, ValueError):
                pass
            return path.read_bytes() == b"old" and set(Path(directory).iterdir()) == {path}

    def rejects_bytearray_without_damage() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"old")
            try:
                module.commit_bytes(path, bytearray(b"new"))
            except (TypeError, ValueError):
                pass
            return path.read_bytes() == b"old" and set(Path(directory).iterdir()) == {path}

    def leaves_no_staging_after_success() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            module.commit_bytes(path, b"new")
            return set(Path(directory).iterdir()) == {path}

    def commits_from_same_directory_staging() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            destination = root / "blob.bin"
            destination.write_bytes(b"old")
            renames: list[tuple[Any, ...]] = []

            def record(event: str, args: tuple[Any, ...]) -> None:
                if event == "os.rename":
                    renames.append(args)

            sys.addaudithook(record)
            module.commit_bytes(destination, b"new")
            for source, target, *_dir_fds in renames:
                source_path = Path(os.fsdecode(source)).resolve()
                target_path = Path(os.fsdecode(target)).resolve()
                if (
                    source_path != destination
                    and source_path.parent == root
                    and target_path == destination
                ):
                    return destination.read_bytes() == b"new"
            return False

    def cleans_staging_after_commit_failure() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "occupied"
            destination.mkdir()
            before = set(root.iterdir())
            try:
                module.commit_bytes(destination, b"new")
            except (OSError, ValueError):
                pass
            return set(root.iterdir()) == before and destination.is_dir()

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def archive_target_v2(module: Any) -> dict[str, bool]:
    """Validate archive paths across POSIX, Windows, and symlink spellings."""

    def maps_nested_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            return Path(module.archive_target(root, "images/icon.png")) == root / "images/icon.png"

    def blocks_dotdot_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.archive_target(Path(directory), "../../escape")
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def blocks_absolute_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.archive_target(Path(directory), "/etc/passwd")
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def blocks_root_alias() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            for member in ("", "."):
                try:
                    module.archive_target(Path(directory), member)
                except (TypeError, ValueError, PermissionError):
                    continue
                return False
            return True

    def blocks_backslash_parent() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.archive_target(Path(directory), r"..\escape.txt")
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def blocks_windows_drive() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.archive_target(Path(directory), r"C:\escape.txt")
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def blocks_linked_directory() -> bool:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "linked").symlink_to(outside, target_is_directory=True)
            try:
                module.archive_target(root, "linked/file")
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def blocks_linked_file() -> bool:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            outside_file = Path(outside) / "secret.txt"
            outside_file.write_text("secret", encoding="utf-8")
            (root / "linked.txt").symlink_to(outside_file)
            try:
                module.archive_target(root, "linked.txt")
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def migration_guard_v2(module: Any) -> dict[str, bool]:
    """Require rollback after exceptions in every started migration phase."""

    def invoke(
        *, check_result: bool = True, verify_result: bool = True,
        failure: str | None = None,
    ) -> list[str]:
        events: list[str] = []

        def callback(name: str, result: bool | None = None) -> Callable[[str], bool | None]:
            def run(_plan: str) -> bool | None:
                events.append(name)
                if failure == name:
                    raise RuntimeError(f"{name} failed")
                return result

            return run

        try:
            module.apply_migration(
                "m1", callback("check", check_result), callback("begin"),
                callback("verify", verify_result), callback("commit"), callback("abort"),
            )
        except Exception:
            pass
        return events

    def commits_verified_change() -> bool:
        return invoke() == ["check", "begin", "verify", "commit"]

    def failed_check_has_no_side_effect() -> bool:
        return invoke(check_result=False) == ["check"]

    def check_exception_has_no_side_effect() -> bool:
        return invoke(failure="check") == ["check"]

    def failed_verify_aborts_once() -> bool:
        return invoke(verify_result=False) == ["check", "begin", "verify", "abort"]

    def begin_exception_aborts_once() -> bool:
        return invoke(failure="begin") == ["check", "begin", "abort"]

    def verify_exception_aborts_once() -> bool:
        return invoke(failure="verify") == ["check", "begin", "verify", "abort"]

    def commit_exception_aborts_once() -> bool:
        return invoke(failure="commit") == ["check", "begin", "verify", "commit", "abort"]

    checks = (
        commits_verified_change,
        failed_check_has_no_side_effect,
        check_exception_has_no_side_effect,
        failed_verify_aborts_once,
        begin_exception_aborts_once,
        verify_exception_aborts_once,
        commit_exception_aborts_once,
    )
    return {function.__name__: attempt(function) for function in checks}


def audit_events_v2(module: Any) -> dict[str, bool]:
    """Require honest statuses and immutable JSON-ready event snapshots."""

    def records_exact_request_and_outcome() -> bool:
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

    def snapshots_nested_values() -> bool:
        input_data = {"items": [{"id": 1}]}
        detail = {"result": ["kept"]}
        events = module.audit_events("update", input_data, "ok", detail)
        input_data["items"][0]["id"] = 9
        detail["result"].append("changed")
        return (
            events[0]["input"] == {"items": [{"id": 1}]}
            and events[1]["detail"] == {"result": ["kept"]}
        )

    def rejects_unknown_status() -> bool:
        try:
            module.audit_events("fetch", {}, "success", {})
        except (TypeError, ValueError):
            return True
        return False

    def rejects_non_string_operation() -> bool:
        try:
            module.audit_events(object(), {}, "ok", {})
        except (TypeError, ValueError):
            return True
        return False

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def chunk_install_eval_v2(module: Any) -> dict[str, bool]:
    """Evaluate streaming atomic installation through a held-out interface."""

    def assembles_chunks() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.bin"
            module.install_payload(destination, [b"header", b"", b"\x00body"])
            return destination.read_bytes() == b"header\x00body"

    def accepts_empty_stream() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.bin"
            destination.write_bytes(b"old")
            module.install_payload(destination, [])
            return destination.read_bytes() == b""

    def preserves_on_late_invalid_chunk() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.bin"
            destination.write_bytes(b"old")

            def chunks() -> Iterator[bytes | bytearray]:
                yield b"partial"
                yield bytearray(b"invalid")

            try:
                module.install_payload(destination, chunks())
            except (TypeError, ValueError):
                pass
            return (
                destination.read_bytes() == b"old"
                and set(Path(directory).iterdir()) == {destination}
            )

    def preserves_on_source_exception() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.bin"
            destination.write_bytes(b"old")

            def chunks() -> Iterator[bytes]:
                yield b"partial"
                raise RuntimeError("source failed")

            try:
                module.install_payload(destination, chunks())
            except RuntimeError:
                pass
            return (
                destination.read_bytes() == b"old"
                and set(Path(directory).iterdir()) == {destination}
            )

    def commits_from_sibling_staging() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            destination = root / "artifact.bin"
            destination.write_bytes(b"old")
            renames: list[tuple[Any, ...]] = []

            def record(event: str, args: tuple[Any, ...]) -> None:
                if event == "os.rename":
                    renames.append(args)

            sys.addaudithook(record)
            module.install_payload(destination, [b"new"])
            return destination.read_bytes() == b"new" and any(
                Path(os.fsdecode(source)).resolve().parent == root
                and Path(os.fsdecode(source)).resolve() != destination
                and Path(os.fsdecode(target)).resolve() == destination
                for source, target, *_dir_fds in renames
            )

    def cleans_after_commit_failure() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "occupied"
            destination.mkdir()
            before = set(root.iterdir())
            try:
                module.install_payload(destination, [b"new"])
            except (OSError, ValueError):
                pass
            return set(root.iterdir()) == before and destination.is_dir()

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def archive_plan_eval_v2(module: Any) -> dict[str, bool]:
    """Evaluate all-or-nothing planning for a batch of archive members."""

    def maps_batch() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            targets = module.plan_archive(root, ["icons/a.png", "docs/readme.txt"])
            return targets == [root / "icons/a.png", root / "docs/readme.txt"]

    def rejects_parent_in_batch() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.plan_archive(Path(directory), ["safe.txt", "../../escape"])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_absolute_in_batch() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.plan_archive(Path(directory), ["safe.txt", "/etc/passwd"])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_portable_aliases() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            unsafe = ("", ".", r"..\escape.txt", r"C:\escape.txt")
            for member in unsafe:
                try:
                    module.plan_archive(Path(directory), ["safe.txt", member])
                except (TypeError, ValueError, PermissionError):
                    continue
                return False
            return True

    def rejects_linked_escape() -> bool:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "linked").symlink_to(outside, target_is_directory=True)
            try:
                module.plan_archive(root, ["safe.txt", "linked/escape.txt"])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_colliding_destinations() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.plan_archive(Path(directory), ["same//file.txt", "same/file.txt"])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def change_transaction_eval_v2(module: Any) -> dict[str, bool]:
    """Evaluate compensation after exceptions through a deployment interface."""

    def invoke(
        *, precheck_result: bool = True, validation_result: bool = True,
        failure: str | None = None,
    ) -> list[str]:
        events: list[str] = []

        def callback(name: str, result: bool | None = None) -> Callable[[str], bool | None]:
            def run(_change: str) -> bool | None:
                events.append(name)
                if failure == name:
                    raise RuntimeError(f"{name} failed")
                return result

            return run

        try:
            module.deploy_change(
                "c1", callback("precheck", precheck_result), callback("stage"),
                callback("validate", validation_result), callback("finalize"),
                callback("revert"),
            )
        except Exception:
            pass
        return events

    def successful_order() -> bool:
        return invoke() == ["precheck", "stage", "validate", "finalize"]

    def rejected_precheck_stops() -> bool:
        return invoke(precheck_result=False) == ["precheck"]

    def precheck_exception_stops() -> bool:
        return invoke(failure="precheck") == ["precheck"]

    def stage_exception_reverts_once() -> bool:
        return invoke(failure="stage") == ["precheck", "stage", "revert"]

    def failed_validation_reverts_once() -> bool:
        return invoke(validation_result=False) == ["precheck", "stage", "validate", "revert"]

    def validation_exception_reverts_once() -> bool:
        return invoke(failure="validate") == ["precheck", "stage", "validate", "revert"]

    def finalize_exception_reverts_once() -> bool:
        return invoke(failure="finalize") == [
            "precheck", "stage", "validate", "finalize", "revert"
        ]

    checks = (
        successful_order,
        rejected_precheck_stops,
        precheck_exception_stops,
        stage_exception_reverts_once,
        failed_validation_reverts_once,
        validation_exception_reverts_once,
        finalize_exception_reverts_once,
    )
    return {function.__name__: attempt(function) for function in checks}


def operation_record_eval_v2(module: Any) -> dict[str, bool]:
    """Evaluate an honest immutable record through a held-out envelope schema."""

    def exact_success_envelope() -> bool:
        record = module.make_operation_record("fetch", {"id": 7}, True, {"value": "ok"})
        return record == {
            "action": "fetch",
            "request": {"id": 7},
            "outcome": {"ok": True, "result": {"value": "ok"}},
        }

    def exact_failure_envelope() -> bool:
        record = module.make_operation_record("delete", {}, False, {"error": "denied"})
        return record == {
            "action": "delete",
            "request": {},
            "outcome": {"ok": False, "result": {"error": "denied"}},
        }

    def snapshots_nested_values() -> bool:
        request = {"items": [{"id": 1}]}
        result = {"kept": ["a"]}
        record = module.make_operation_record("update", request, True, result)
        request["items"][0]["id"] = 2
        result["kept"].append("b")
        return (
            record["request"] == {"items": [{"id": 1}]}
            and record["outcome"]["result"] == {"kept": ["a"]}
        )

    def rejects_truthy_non_boolean() -> bool:
        try:
            module.make_operation_record("fetch", {}, 1, {})
        except (TypeError, ValueError):
            return True
        return False

    def rejects_invalid_action() -> bool:
        for action in ("", object()):
            try:
                module.make_operation_record(action, {}, True, {})
            except (TypeError, ValueError):
                continue
            return False
        return True

    def serializes_to_json() -> bool:
        json.dumps(module.make_operation_record("read", [1], True, ["ok"]))
        return True

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def bundle_commit_eval_v3(module: Any) -> dict[str, bool]:
    """Evaluate atomic two-file bundle commits through a held-out interface."""

    def writes_bundle() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            destination.mkdir()
            module.commit_bundle(destination, b"data\x00payload", {"version": 1, "files": ["a.bin"]})
            return (
                (destination / "payload.bin").read_bytes() == b"data\x00payload"
                and json.loads((destination / "manifest.json").read_text())
                == {"version": 1, "files": ["a.bin"]}
            )

    def replaces_old_bundle() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            destination.mkdir()
            (destination / "payload.bin").write_bytes(b"old")
            (destination / "manifest.json").write_text('{"old": true}')
            module.commit_bundle(destination, b"new", {"new": True})
            return (
                (destination / "payload.bin").read_bytes() == b"new"
                and json.loads((destination / "manifest.json").read_text()) == {"new": True}
            )

    def preserves_on_invalid_manifest() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            destination.mkdir()
            (destination / "payload.bin").write_bytes(b"old")
            (destination / "manifest.json").write_text('{"old": true}')
            before = set(destination.iterdir())
            try:
                module.commit_bundle(destination, b"new", "not-a-dict")
            except (TypeError, ValueError):
                pass
            return (
                (destination / "payload.bin").read_bytes() == b"old"
                and set(destination.iterdir()) == before
            )

    def preserves_on_invalid_payload() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            destination.mkdir()
            (destination / "payload.bin").write_bytes(b"old")
            (destination / "manifest.json").write_text('{"old": true}')
            before = set(destination.iterdir())
            try:
                module.commit_bundle(destination, object(), {"new": True})
            except (TypeError, ValueError):
                pass
            return (
                (destination / "payload.bin").read_bytes() == b"old"
                and set(destination.iterdir()) == before
            )

    def cleans_staging_after_commit_failure() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "occupied"
            destination.write_bytes(b"file")  # destination is a file, not a directory
            before = set(root.iterdir())
            try:
                module.commit_bundle(destination, b"new", {"new": True})
            except (OSError, ValueError):
                pass
            return set(root.iterdir()) == before

    def commits_from_sibling_staging() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            destination = root / "bundle"
            destination.mkdir()
            renames: list[tuple[Any, ...]] = []

            def record(event: str, args: tuple[Any, ...]) -> None:
                if event == "os.rename":
                    renames.append(args)

            sys.addaudithook(record)
            module.commit_bundle(destination, b"new", {"new": True})
            return len(renames) == 2 and all(
                Path(os.fsdecode(source)).resolve().parent == destination
                and Path(os.fsdecode(target)).resolve().parent == destination
                for source, target, *_dir_fds in renames
            )

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def extract_manifest_eval_v3(module: Any) -> dict[str, bool]:
    """Evaluate whole-batch manifest validation through a held-out interface."""

    def maps_all_members() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            digest = "a" * 64
            targets = module.extract_manifest(
                root,
                [{"name": "icons/a.png", "sha256": digest}, {"name": "docs/readme.txt", "sha256": digest}],
            )
            return targets == [root / "icons/a.png", root / "docs/readme.txt"]

    def rejects_traversal_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.extract_manifest(Path(directory), [{"name": "../escape", "sha256": "a" * 64}])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_absolute_member() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.extract_manifest(Path(directory), [{"name": "/etc/passwd", "sha256": "a" * 64}])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_linked_escape() -> bool:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "linked").symlink_to(outside, target_is_directory=True)
            try:
                module.extract_manifest(root, [{"name": "linked/secret", "sha256": "a" * 64}])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_colliding_members() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.extract_manifest(
                    Path(directory),
                    [{"name": "same//file.txt", "sha256": "a" * 64}, {"name": "same/file.txt", "sha256": "a" * 64}],
                )
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    def rejects_malformed_digest() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            for bad in ("nothex", "abc", "", 42, None):
                try:
                    module.extract_manifest(Path(directory), [{"name": "ok.txt", "sha256": bad}])
                except (TypeError, ValueError, PermissionError):
                    continue
                return False
            return True

    def rejects_missing_sha256_field() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.extract_manifest(Path(directory), [{"name": "ok.txt"}])
            except (TypeError, ValueError, PermissionError):
                return True
            return False

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def install_payload_train_v2(module: Any) -> dict[str, bool]:
    """Training alias: mirror chunk_install_eval_v2 behavioral checks.

    Teacher answers for the streaming-atomic-install interface prove they
    satisfy the exact held-out semantics (sibling staging, fsync ordering,
    preservation on late invalid chunks, cleanup after commit failure).
    """
    return chunk_install_eval_v2(module)


def plan_archive_train_v2(module: Any) -> dict[str, bool]:
    """Training alias: mirror archive_plan_eval_v2 behavioral checks.

    Teacher answers for the whole-batch archive planning interface prove they
    satisfy the exact held-out semantics (all-or-nothing rejection, portable
    aliases, linked escapes, colliding destinations).
    """
    return archive_plan_eval_v2(module)


def deploy_gate_eval_v3(module: Any) -> dict[str, bool]:
    """Evaluate backup-before-switch deployment with compensation through a held-out interface."""

    def _invoke(*, precheck_ok: bool = True, backup_fails: bool = False,
                switch_fails: bool = False, verify_ok: bool = True) -> list[str]:
        events: list[str] = []

        def callback(name: str, result: bool | None = None) -> Callable[[str], bool | None]:
            def run(_plan: str) -> bool | None:
                events.append(name)
                if name == "backup" and backup_fails:
                    raise RuntimeError("backup failed")
                if name == "switch" and switch_fails:
                    raise RuntimeError("switch failed")
                return result
            return run

        try:
            module.deploy_gate(
                "p1",
                callback("precheck", precheck_ok),
                callback("backup", True),
                callback("switch", True),
                callback("verify", verify_ok),
                callback("revert", True),
            )
        except Exception:
            pass
        return events

    def success_order() -> bool:
        return _invoke() == ["precheck", "backup", "switch", "verify"]

    def failed_precheck_stops() -> bool:
        return _invoke(precheck_ok=False) == ["precheck"]

    def backup_failure_stops_without_revert() -> bool:
        return _invoke(backup_fails=True) == ["precheck", "backup"]

    def switch_failure_reverts_once() -> bool:
        return _invoke(switch_fails=True) == ["precheck", "backup", "switch", "revert"]

    def failed_verify_reverts_once() -> bool:
        return _invoke(verify_ok=False) == ["precheck", "backup", "switch", "verify", "revert"]

    def verify_exception_reverts_once() -> bool:
        events: list[str] = []

        def run(_plan: str) -> bool | None:
            events.append("verify")
            raise RuntimeError("verify failed")

        try:
            module.deploy_gate("p1", lambda _p: events.append("precheck") or True,
                               lambda _p: events.append("backup") or True,
                               lambda _p: events.append("switch") or True,
                               run, lambda _p: events.append("revert") or True)
        except Exception:
            pass
        return events == ["precheck", "backup", "switch", "verify", "revert"]

    return {
        name: attempt(function)
        for name, function in locals().copy().items()
        if callable(function) and not name.startswith("_")
    }


def quarantine_move_eval_v3(module: Any) -> dict[str, bool]:
    """Evaluate all-or-nothing file moves with rollback through a held-out interface."""

    def moves_all() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quarantine = root / "quarantine"
            quarantine.mkdir()
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("one")
            second.write_text("two")
            targets = module.quarantine_move([first, second], quarantine)
            return (
                sorted(path.name for path in targets) == ["first.txt", "second.txt"]
                and (quarantine / "first.txt").read_text() == "one"
                and (quarantine / "second.txt").read_text() == "two"
                and not first.exists() and not second.exists()
            )

    def preserves_on_missing_source() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quarantine = root / "quarantine"
            quarantine.mkdir()
            first = root / "first.txt"
            first.write_text("one")
            try:
                module.quarantine_move([first, root / "missing.txt"], quarantine)
            except (OSError, ValueError):
                pass
            return first.read_text() == "one" and set(quarantine.iterdir()) == set()

    def rejects_preexisting_target() -> bool:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quarantine = root / "quarantine"
            quarantine.mkdir()
            first = root / "first.txt"
            first.write_text("one")
            (quarantine / "first.txt").write_text("existing")
            try:
                module.quarantine_move([first], quarantine)
            except (OSError, ValueError):
                pass
            return (quarantine / "first.txt").read_text() == "existing" and first.exists()

    return {name: attempt(function) for name, function in locals().copy().items() if callable(function)}


def redact_secrets_eval_v3(module: Any) -> dict[str, bool]:
    """Evaluate deterministic secret redaction through a held-out interface."""

    def redacts_every_match() -> bool:
        text = "token sk-abcdefgh12345678 and key AKIA1234567890ABCDEF"
        redacted = module.redact_secrets(text, [r"sk-[A-Za-z0-9]{16,}", r"AKIA[A-Z0-9]{16}"])
        return "sk-abcdefgh12345678" not in redacted and "AKIA1234567890ABCDEF" not in redacted

    def preserves_clean_text() -> bool:
        text = "no secrets here, just a build log"
        return module.redact_secrets(text, [r"sk-[A-Za-z0-9]{16,}"]) == text

    def handles_multiple_patterns() -> bool:
        text = "user alice, token sk-abcdefgh12345678, key AKIA1234567890ABCDEF"
        redacted = module.redact_secrets(text, [r"sk-[A-Za-z0-9]{16,}", r"AKIA[A-Z0-9]{16}"])
        return "alice" in redacted and "sk-abcdefgh12345678" not in redacted and "AKIA1234567890ABCDEF" not in redacted

    def deterministic_output() -> bool:
        text = "token sk-abcdefgh12345678 here"
        pattern = [r"sk-[A-Za-z0-9]{16,}"]
        return module.redact_secrets(text, pattern) == module.redact_secrets(text, pattern)

    def marker_never_contains_secret() -> bool:
        text = "xsk-abcdefgh12345678y"
        redacted = module.redact_secrets(text, [r"sk-[A-Za-z0-9]{16,}"])
        return "sk-abcdefgh12345678" not in redacted

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
    "commit_bytes_train_v2": commit_bytes_v2,
    "archive_target_train_v2": archive_target_v2,
    "migration_guard_train_v2": migration_guard_v2,
    "audit_events_train_v2": audit_events_v2,
    "chunk_install_eval_v2": chunk_install_eval_v2,
    "archive_plan_eval_v2": archive_plan_eval_v2,
    "change_transaction_eval_v2": change_transaction_eval_v2,
    "operation_record_eval_v2": operation_record_eval_v2,
    "install_payload_train_v2": install_payload_train_v2,
    "plan_archive_train_v2": plan_archive_train_v2,
    "bundle_commit_eval_v3": bundle_commit_eval_v3,
    "extract_manifest_eval_v3": extract_manifest_eval_v3,
    "deploy_gate_eval_v3": deploy_gate_eval_v3,
    "quarantine_move_eval_v3": quarantine_move_eval_v3,
    "redact_secrets_eval_v3": redact_secrets_eval_v3,
}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in HARNESSES:
        raise SystemExit(2)
    submission = Path(sys.argv[2]).resolve()
    install_safety_audit_hook(submission)
    module = load(str(submission))
    outcomes = HARNESSES[sys.argv[1]](module)
    sys.stdout.write(json.dumps(outcomes, sort_keys=True))


if __name__ == "__main__":
    main()
