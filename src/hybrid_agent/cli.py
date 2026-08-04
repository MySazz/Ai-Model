"""Command-line interface for the hybrid agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .agent import AgentSession
from .audit import AuditStore
from .experiments import ExperimentError, verify_experiment, write_blocked_baseline
from .evaluation import EvaluationError, evaluate_responses, load_jsonl_objects
from .memory import MemoryStore
from .permissions import PermissionPolicy
from .privacy import PrivacyLevel
from .providers import OfflineProvider, OllamaProvider
from .research import ResearchStore
from .tools import Tool, default_registry
from .training_data import load_jsonl, prepare_dataset, validate_records
from .vision import load_image


def default_state_dir() -> Path:
    configured = os.environ.get("HYBRID_AGENT_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".hybrid-agent"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-agent",
        description="Local-first hybrid AI agent",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Start a one-shot or interactive agent session")
    chat.add_argument("prompt", nargs="?", help="Run one prompt; omit for interactive mode")
    chat.add_argument("--provider", choices=("offline", "ollama"), default="offline")
    chat.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", ""))
    chat.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    chat.add_argument("--workspace", type=Path, default=Path.cwd())
    chat.add_argument("--state-dir", type=Path, default=default_state_dir())
    chat.add_argument("--user-id", default="local")
    chat.add_argument("--workspace-id", default="default")
    chat.add_argument("--no-memory", action="store_true")
    chat.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Attach a workspace PNG, JPEG, or GIF; repeat for multiple images",
    )

    audit = subparsers.add_parser("audit", help="Show recent audit events")
    audit.add_argument("--limit", type=int, default=20)
    audit.add_argument("--state-dir", type=Path, default=default_state_dir())
    audit.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check local configuration")
    doctor.add_argument("--state-dir", type=Path, default=default_state_dir())

    memory = subparsers.add_parser("memory", help="Manage explicit long-term memory")
    memory.add_argument("--state-dir", type=Path, default=default_state_dir())
    memory.add_argument("--user-id", default="local")
    memory.add_argument("--workspace-id", default="default")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)
    memory_add = memory_subparsers.add_parser("add", help="Store a user-approved memory")
    memory_add.add_argument("content")
    memory_add.add_argument("--category", default="fact")
    memory_search = memory_subparsers.add_parser("search", help="Search stored memories")
    memory_search.add_argument("query", nargs="?", default="")
    memory_search.add_argument("--limit", type=int, default=20)

    sources = subparsers.add_parser("sources", help="Manage approved research sources")
    sources.add_argument("--state-dir", type=Path, default=default_state_dir())
    sources.add_argument("--workspace", type=Path, default=Path.cwd())
    sources.add_argument("--user-id", default="local")
    sources.add_argument("--workspace-id", default="default")
    source_subparsers = sources.add_subparsers(dest="source_command", required=True)
    source_add = source_subparsers.add_parser("add", help="Ingest a workspace text file")
    source_add.add_argument("path", type=Path)
    source_add.add_argument("--title")
    source_list = source_subparsers.add_parser("list", help="List research sources")
    source_list.add_argument("--limit", type=int, default=50)
    source_search = source_subparsers.add_parser("search", help="Search research sources")
    source_search.add_argument("query")
    source_search.add_argument("--limit", type=int, default=20)

    dataset = subparsers.add_parser("dataset", help="Validate and prepare training datasets")
    dataset.add_argument("--workspace", type=Path, default=Path.cwd())
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_subparsers.add_parser(
        "validate", help="Validate conversational JSONL"
    )
    dataset_validate.add_argument("paths", nargs="+", type=Path)
    dataset_validate.add_argument("--json", action="store_true")
    dataset_prepare = dataset_subparsers.add_parser(
        "prepare", help="Validate, normalize, deduplicate, and split JSONL"
    )
    dataset_prepare.add_argument("paths", nargs="+", type=Path)
    dataset_prepare.add_argument("--output", type=Path, required=True)
    dataset_prepare.add_argument("--seed", default="hybrid-agent-v1")
    dataset_prepare.add_argument("--train-ratio", type=float, default=0.8)
    dataset_prepare.add_argument("--validation-ratio", type=float, default=0.1)

    experiment = subparsers.add_parser(
        "experiment", help="Verify and run pinned training experiments"
    )
    experiment.add_argument("--workspace", type=Path, default=Path.cwd())
    experiment.add_argument(
        "--config", type=Path, default=Path("training/configs/qwen3-1.7b-qlora-v1.toml")
    )
    experiment_subparsers = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    experiment_subparsers.add_parser("verify", help="Verify all frozen input hashes")
    blocked = experiment_subparsers.add_parser(
        "record-blocked-baseline", help="Record why the pinned baseline could not run"
    )
    blocked.add_argument("--output", type=Path, required=True)
    blocked.add_argument("--reason", required=True)
    blocked.add_argument("--evidence", action="append", default=[])

    evaluation = subparsers.add_parser(
        "evaluate", help="Machine-score model responses against an automated suite"
    )
    evaluation.add_argument("--workspace", type=Path, default=Path.cwd())
    evaluation.add_argument("--suite", type=Path, required=True)
    evaluation.add_argument("--responses", type=Path, required=True)
    evaluation.add_argument("--output", type=Path)
    return parser


def make_provider(name: str, model: str, ollama_url: str):
    if name == "offline":
        return OfflineProvider()
    if not model:
        raise ValueError("Ollama requires --model or the OLLAMA_MODEL environment variable.")
    return OllamaProvider(model=model, base_url=ollama_url)


def prompt_for_tool_approval(
    tool: Tool,
    arguments: dict[str, object],
    reason: str,
) -> bool:
    if not sys.stdin.isatty():
        return False
    argument_summary = ", ".join(
        f"{key}=<{len(str(value))} chars>" for key, value in arguments.items()
    )
    risk = tool.assess_risk(arguments)
    print("\nApproval required")
    print(f"Action: {tool.name}")
    print(f"Risk: {risk.name.lower()}")
    print(f"Arguments: {argument_summary or 'none'}")
    print(f"Reason: {reason}")
    try:
        preview = tool.preview(arguments)
    except Exception as exc:
        print(f"Preview failed: {type(exc).__name__}: {exc}")
        return False
    if preview:
        print("Preview:")
        print(preview)
    return input("Allow this action once? [y/N] ").strip().lower() in {"y", "yes"}


def prompt_for_cloud_approval(level: PrivacyLevel, reasons: tuple[str, ...]) -> bool:
    if not sys.stdin.isatty():
        return False
    print("\nCloud disclosure approval required")
    print(f"Privacy level: {level.name.lower()}")
    print(f"Reason: {', '.join(reasons)}")
    answer = input("Send this content to the configured cloud model once? [y/N] ")
    return answer.strip().lower() in {
        "y",
        "yes",
    }


def run_chat(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"Workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    try:
        provider = make_provider(args.provider, args.model, args.ollama_url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    research = ResearchStore(args.state_dir / "research.db")
    session = AgentSession(
        provider=provider,
        tools=default_registry(
            workspace,
            research,
            user_id=args.user_id,
            workspace_id=args.workspace_id,
        ),
        audit=AuditStore(args.state_dir / "agent.db"),
        policy=PermissionPolicy(),
        memory=None if args.no_memory else MemoryStore(args.state_dir / "memory.db"),
        user_id=args.user_id,
        workspace_id=args.workspace_id,
        approval_handler=prompt_for_tool_approval,
        cloud_approval_handler=prompt_for_cloud_approval,
    )
    try:
        images = tuple(load_image(path, workspace=workspace) for path in args.image)
    except (OSError, ValueError, PermissionError) as exc:
        print(f"Image error: {exc}", file=sys.stderr)
        return 2
    if args.prompt:
        try:
            print(session.run(args.prompt, images=images))
        except (RuntimeError, KeyError, PermissionError) as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            return 1
        return 0

    print(f"Hybrid Agent ({provider.name}) — type /quit to exit")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt in {"/quit", "/exit"}:
            return 0
        if not prompt:
            continue
        try:
            print(f"agent> {session.run(prompt, images=images)}")
        except (RuntimeError, KeyError, PermissionError) as exc:
            print(f"Agent error: {exc}", file=sys.stderr)


def run_audit(args: argparse.Namespace) -> int:
    events = AuditStore(args.state_dir / "agent.db").recent(max(1, args.limit))
    for event in events:
        data = {
            "id": event.id,
            "created_at": event.created_at,
            "session_id": event.session_id,
            "type": event.event_type,
            "tool": event.tool_name,
            "risk": event.risk,
            "approved": event.approved,
            "status": event.status,
            "details": event.details,
        }
        if args.json:
            print(json.dumps(data, sort_keys=True))
        else:
            target = f" tool={event.tool_name}" if event.tool_name else ""
            print(f"{event.id} {event.created_at} {event.event_type}{target} [{event.status}]")
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    state_dir = args.state_dir.resolve()
    AuditStore(state_dir / "agent.db")
    checks = [
        ("Python", sys.version.split()[0], True),
        ("State directory", str(state_dir), os.access(state_dir, os.W_OK)),
        ("OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "not configured"), True),
    ]
    for name, value, healthy in checks:
        print(f"{'ok' if healthy else 'error':5} {name}: {value}")
    return 0 if all(check[2] for check in checks) else 1


def run_memory(args: argparse.Namespace) -> int:
    store = MemoryStore(args.state_dir / "memory.db")
    if args.memory_command == "add":
        memory_id = store.add(
            args.content,
            category=args.category,
            user_id=args.user_id,
            workspace_id=args.workspace_id,
        )
        print(f"Stored memory {memory_id}.")
        return 0
    for memory in store.search(
        args.query,
        limit=args.limit,
        user_id=args.user_id,
        workspace_id=args.workspace_id,
    ):
        print(f"{memory.id} [{memory.category}] {memory.content}")
    return 0


def run_sources(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    store = ResearchStore(args.state_dir / "research.db")
    if args.source_command == "add":
        path = args.path if args.path.is_absolute() else workspace / args.path
        try:
            source_id = store.add_workspace_file(
                path,
                workspace=workspace,
                title=args.title,
                user_id=args.user_id,
                workspace_id=args.workspace_id,
            )
        except (OSError, UnicodeError, ValueError, PermissionError) as exc:
            print(f"Source error: {exc}", file=sys.stderr)
            return 2
        print(f"Stored research source {source_id}.")
        return 0
    if args.source_command == "list":
        for source in store.list(
            user_id=args.user_id,
            workspace_id=args.workspace_id,
            limit=args.limit,
        ):
            print(f"{source.id} {source.title} [{source.uri}] sha256={source.sha256}")
        return 0
    for match in store.search(
        args.query,
        user_id=args.user_id,
        workspace_id=args.workspace_id,
        limit=args.limit,
    ):
        print(
            f"{match.source_id} {match.uri}:{match.line_number} "
            f"sha256={match.sha256} {match.excerpt}"
        )
    return 0


def run_dataset(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    paths: list[Path] = []
    try:
        for path in args.paths:
            resolved = path.resolve()
            if resolved != workspace and workspace not in resolved.parents:
                raise PermissionError(f"Dataset path escapes the workspace: {path}")
            paths.append(resolved)
    except (OSError, PermissionError) as exc:
        print(f"Dataset access failed: {exc}", file=sys.stderr)
        return 2

    if args.dataset_command == "validate":
        records, load_issues = load_jsonl(paths)
        report = validate_records(records, load_issues)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(
                f"records={report.records} errors={report.errors} "
                f"warnings={report.warnings} valid={str(report.valid).lower()}"
            )
            for issue in report.issues:
                record = f" record={issue.record_id}" if issue.record_id else ""
                print(
                    f"{issue.severity}: {issue.source}:{issue.line}{record} "
                    f"[{issue.code}] {issue.message}"
                )
        return 0 if report.valid else 1

    try:
        output = args.output.resolve()
        if output != workspace and workspace not in output.parents:
            raise PermissionError(f"Dataset output escapes the workspace: {args.output}")
        manifest = prepare_dataset(
            paths,
            output,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )
    except (OSError, UnicodeError, ValueError, PermissionError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(
        f"Prepared train={counts['train']} validation={counts['validation']} "
        f"test={counts['test']} in {output}"
    )
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    config = args.config if args.config.is_absolute() else workspace / args.config
    try:
        verified = verify_experiment(config, workspace=workspace)
        if args.experiment_command == "verify":
            print(
                f"Verified {verified.config['experiment_id']}: "
                f"manifest={verified.manifest_sha256} evaluation={verified.evaluation_sha256}"
            )
            return 0
        output = args.output if args.output.is_absolute() else workspace / args.output
        resolved_output = output.resolve()
        if resolved_output != workspace and workspace not in resolved_output.parents:
            raise ExperimentError(f"Baseline output escapes workspace: {output}")
        payload = write_blocked_baseline(
            verified, resolved_output, reason=args.reason, evidence=args.evidence
        )
        print(f"Recorded {payload['status']} baseline result in {resolved_output}")
        return 0
    except ExperimentError as exc:
        print(f"Experiment refused: {exc}", file=sys.stderr)
        return 1


def run_evaluation(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    try:
        paths = []
        for path in (args.suite, args.responses):
            resolved = (path if path.is_absolute() else workspace / path).resolve()
            if resolved != workspace and workspace not in resolved.parents:
                raise EvaluationError(f"Evaluation path escapes workspace: {path}")
            paths.append(resolved)
        report = evaluate_responses(
            load_jsonl_objects(paths[0]), load_jsonl_objects(paths[1])
        )
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = (args.output if args.output.is_absolute() else workspace / args.output).resolve()
            if output != workspace and workspace not in output.parents:
                raise EvaluationError(f"Evaluation output escapes workspace: {args.output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if not report["critical_failures"] else 1
    except (OSError, UnicodeError, EvaluationError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "chat":
        return run_chat(args)
    if args.command == "audit":
        return run_audit(args)
    if args.command == "doctor":
        return run_doctor(args)
    if args.command == "memory":
        return run_memory(args)
    if args.command == "sources":
        return run_sources(args)
    if args.command == "dataset":
        return run_dataset(args)
    if args.command == "experiment":
        return run_experiment(args)
    if args.command == "evaluate":
        return run_evaluation(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
