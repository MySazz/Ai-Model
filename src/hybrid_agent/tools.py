"""Tool registry with workspace boundaries and declared risk."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .permissions import RiskLevel
from .research import ResearchStore


ToolHandler = Callable[[dict[str, Any]], str]
RiskAssessor = Callable[[dict[str, Any]], RiskLevel]
Previewer = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: RiskLevel | RiskAssessor
    parameters: dict[str, Any]
    handler: ToolHandler
    previewer: Previewer | None = None

    def assess_risk(self, arguments: dict[str, Any]) -> RiskLevel:
        return self.risk(arguments) if callable(self.risk) else self.risk

    def preview(self, arguments: dict[str, Any]) -> str | None:
        return self.previewer(arguments) if self.previewer else None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk_level": "dynamic" if callable(self.risk) else int(self.risk),
        }


class ToolRegistry:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def resolve_path(self, raw_path: str) -> Path:
        candidate = (self.workspace / raw_path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise PermissionError("Path escapes the configured workspace.")
        return candidate


def default_registry(
    workspace: Path,
    research: ResearchStore | None = None,
    *,
    user_id: str = "local",
    workspace_id: str = "default",
    sandbox: str = "none",
) -> ToolRegistry:
    registry = ToolRegistry(workspace)

    def list_files(arguments: dict[str, Any]) -> str:
        path = registry.resolve_path(str(arguments.get("path", ".")))
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        return "\n".join(
            str(item.relative_to(registry.workspace))
            for item in sorted(path.iterdir(), key=lambda item: item.name)
        )

    def read_file(arguments: dict[str, Any]) -> str:
        path = registry.resolve_path(str(arguments["path"]))
        try:
            max_chars = min(int(arguments.get("max_chars", 20_000)), 100_000)
        except (TypeError, ValueError):
            max_chars = 20_000
        try:
            return path.read_text(encoding="utf-8")[:max_chars]
        except UnicodeDecodeError:
            raise ValueError(f"File '{path.name}' is binary or not valid UTF-8.")

    def search_text(arguments: dict[str, Any]) -> str:
        path = registry.resolve_path(str(arguments.get("path", ".")))
        query = str(arguments["query"])
        if not query:
            raise ValueError("Search query cannot be empty.")
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches: list[str] = []
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if len(matches) >= 200 or not candidate.is_file():
                continue
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in candidate.parts):
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    relative = candidate.relative_to(registry.workspace)
                    matches.append(f"{relative}:{line_number}:{line[:500]}")
                    if len(matches) >= 200:
                        break
        return "\n".join(matches)

    def git_status(arguments: dict[str, Any]) -> str:
        path = registry.resolve_path(str(arguments.get("path", ".")))
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            return f"Error ({result.returncode}):\n{result.stderr[:20000]}"
        return result.stdout[:20000]

    def write_file(arguments: dict[str, Any]) -> str:
        path = registry.resolve_path(str(arguments["path"]))
        content = str(arguments["content"])
        if path.exists():
            raise FileExistsError("Refusing to overwrite an existing file.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Could not create parent directory: {exc}")
        path.write_text(content, encoding="utf-8")
        return f"Created {path.relative_to(registry.workspace)} ({len(content)} characters)."

    def command_risk(arguments: dict[str, Any]) -> RiskLevel:
        argv = arguments.get("argv")
        valid_argv = (
            isinstance(argv, list)
            and bool(argv)
            and all(isinstance(item, str) for item in argv)
        )
        if not valid_argv:
            return RiskLevel.PROHIBITED
        executable = argv[0]
        if executable == "git" and len(argv) >= 2:
            if argv[1] in {"status", "diff", "log"}:
                return RiskLevel.OBSERVE
            return RiskLevel.PROHIBITED
        if executable == "rg":
            return RiskLevel.OBSERVE
        if executable in {"python", "python3"} and argv[1:3] in (
            ["-m", "pytest"],
            ["-m", "compileall"],
        ):
            return RiskLevel.APPROVAL
        return RiskLevel.PROHIBITED

    def validate_command(argv: list[str]) -> list[str]:
        risk = command_risk({"argv": argv})
        if risk is RiskLevel.PROHIBITED:
            raise PermissionError("Command is outside the initial execution allowlist.")
        if argv[0] == "git":
            subcommand = argv[1]
            allowed_arguments = {
                "status": {"--short", "--branch", "--porcelain", "--untracked-files=no"},
                "diff": {"--stat", "--check", "--cached", "--no-ext-diff"},
                "log": {"--oneline", "--decorate", "--all"},
            }
            for item in argv[2:]:
                if subcommand == "log" and (item.isdigit() or item in {"-n", "--max-count"}):
                    continue
                if item not in allowed_arguments[subcommand]:
                    message = f"Argument is not allowed for git {subcommand}: {item}"
                    raise PermissionError(message)
        elif argv[0] == "rg":
            if len(argv) not in {2, 3} or any(item.startswith("-") for item in argv[1:]):
                raise PermissionError(
                    "rg accepts only a literal query and optional workspace path."
                )
            if len(argv) == 3:
                registry.resolve_path(argv[2])
        else:
            for item in argv[3:]:
                if item.startswith("-"):
                    if item not in {"-q", "--quiet", "-x", "--exit-zero"}:
                        raise PermissionError(f"Command option is not allowed: {item}")
                else:
                    registry.resolve_path(item)
        return argv

    def run_command(arguments: dict[str, Any]) -> str:
        raw_argv = arguments.get("argv")
        if not isinstance(raw_argv, list) or not all(isinstance(item, str) for item in raw_argv):
            raise ValueError("argv must be a non-empty array of strings.")
        argv = validate_command(raw_argv)
        cwd = registry.resolve_path(str(arguments.get("cwd", ".")))
        try:
            timeout = min(max(float(arguments.get("timeout", 30)), 1), 60)
        except (TypeError, ValueError):
            timeout = 30.0
        env_path = os.environ.get("PATH", "")
        safe_path = f"{env_path}:/usr/local/bin:/usr/bin:/bin" if env_path else "/usr/local/bin:/usr/bin:/bin"
        
        if sandbox == "docker":
            argv = [
                "docker", "run", "--rm", "-i",
                "--network", "none",
                "-v", f"{workspace.resolve()}:/workspace",
                "-w", f"/workspace/{cwd.relative_to(workspace.resolve()) if cwd.is_relative_to(workspace.resolve()) else ''}",
                "python:3.12-slim"
            ] + argv
            cwd = None  # Docker handles CWD
            
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={"PATH": safe_path, "LANG": "C.UTF-8"},
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if len(output) > 20000:
            output = output[:20000] + "\n...[truncated due to length]"
        return f"exit_code={result.returncode}\n{output}"

    def search_sources(arguments: dict[str, Any]) -> str:
        if research is None:
            raise RuntimeError("Research storage is not configured.")
        try:
            limit = min(int(arguments.get("limit", 20)), 50)
        except (TypeError, ValueError):
            limit = 20
        matches = research.search(
            str(arguments["query"]),
            user_id=user_id,
            workspace_id=workspace_id,
            limit=limit,
        )
        return "\n".join(
            (
                f"[source_id={match.source_id}; {match.uri}:{match.line_number}; "
                f"sha256={match.sha256}] {match.excerpt}"
            )
            for match in matches
        )

    def patch_preview(arguments: dict[str, Any]) -> str:
        path = registry.resolve_path(str(arguments["path"]))
        current = path.read_text(encoding="utf-8")
        old_text = str(arguments["old_text"])
        new_text = str(arguments["new_text"])
        if not old_text:
            raise ValueError("old_text cannot be empty.")
        if current.count(old_text) != 1:
            raise ValueError("old_text must match exactly once in the current file.")
        updated = current.replace(old_text, new_text, 1)
        relative = str(path.relative_to(registry.workspace))
        digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
        diff = "".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
        return f"current_sha256={digest}\n{diff}"[:100_000]

    def apply_patch(arguments: dict[str, Any]) -> str:
        preview = patch_preview(arguments)
        path = registry.resolve_path(str(arguments["path"]))
        current = path.read_text(encoding="utf-8")
        updated = current.replace(
            str(arguments["old_text"]),
            str(arguments["new_text"]),
            1,
        )
        digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
        backup = registry.resolve_path(f".hybrid-agent/backups/{digest}.bak")
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(current, encoding="utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
            raise
        return (
            f"Applied approved patch to {path.relative_to(registry.workspace)}. "
            f"Backup: {backup.relative_to(registry.workspace)}\n{preview}"
        )

    registry.register(
        Tool(
            name="list_files",
            description="List files in a directory inside the configured workspace.",
            risk=RiskLevel.OBSERVE,
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            handler=list_files,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file inside the configured workspace.",
            risk=RiskLevel.OBSERVE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=read_file,
        )
    )
    registry.register(
        Tool(
            name="search_text",
            description="Search text files inside the workspace for a literal string.",
            risk=RiskLevel.OBSERVE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
            handler=search_text,
        )
    )
    registry.register(
        Tool(
            name="git_status",
            description="Inspect Git branch and working-tree status without modifying it.",
            risk=RiskLevel.OBSERVE,
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            handler=git_status,
        )
    )
    registry.register(
        Tool(
            name="write_new_file",
            description="Create a new UTF-8 file; never overwrites an existing file.",
            risk=RiskLevel.APPROVAL,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
        )
    )
    registry.register(
        Tool(
            name="run_command",
            description=(
                "Run a bounded allowlisted command without a shell. Read-only Git and rg "
                "commands are automatic; test and compile commands require approval."
            ),
            risk=command_risk,
            parameters={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["argv"],
            },
            handler=run_command,
        )
    )
    registry.register(
        Tool(
            name="search_research_sources",
            description=(
                "Search user-approved research sources and return line-level provenance."
            ),
            risk=RiskLevel.OBSERVE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=search_sources,
        )
    )
    registry.register(
        Tool(
            name="apply_text_patch",
            description=(
                "Replace text that matches exactly once in an existing file. "
                "Shows a unified diff before approval and creates a content-addressed backup."
            ),
            risk=RiskLevel.APPROVAL,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
            handler=apply_patch,
            previewer=patch_preview,
        )
    )
    return registry
