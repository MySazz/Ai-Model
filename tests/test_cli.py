import json
from pathlib import Path

from hybrid_agent.cli import main


def test_doctor_and_offline_chat_commands(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state"
    assert main(["doctor", "--state-dir", str(state)]) == 0
    assert "State directory" in capsys.readouterr().out

    assert main(
        [
            "chat",
            "hello",
            "--provider",
            "offline",
            "--workspace",
            str(tmp_path),
            "--state-dir",
            str(state),
            "--no-memory",
        ]
    ) == 0
    assert "I received: 'hello'" in capsys.readouterr().out


def test_memory_and_source_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    state = tmp_path / "state"
    monkeypatch.setattr("hybrid_agent.memory._get_embedding", lambda _text: [])
    memory_args = [
        "--state-dir",
        str(state),
        "--user-id",
        "test-user",
        "--workspace-id",
        "test-workspace",
    ]
    assert main(["memory", *memory_args, "add", "Remember this"]) == 0
    capsys.readouterr()
    assert main(["memory", *memory_args, "search", "Remember"]) == 0
    assert "Remember this" in capsys.readouterr().out

    document = tmp_path / "source.txt"
    document.write_text("Traceable evidence\n", encoding="utf-8")
    source_args = [
        "--state-dir",
        str(state),
        "--workspace",
        str(tmp_path),
        "--user-id",
        "test-user",
        "--workspace-id",
        "test-workspace",
    ]
    assert main(["sources", *source_args, "add", "source.txt"]) == 0
    capsys.readouterr()
    assert main(["sources", *source_args, "search", "evidence"]) == 0
    assert "source.txt:1" in capsys.readouterr().out


def test_dataset_validate_command(tmp_path: Path, capsys) -> None:
    source = tmp_path / "records.jsonl"
    record = {
        "id": "cli-record",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Verified answer"},
        ],
        "metadata": {
            "source": "CLI test",
            "license": "proprietary-approved",
            "category": "coding",
            "reviewed": True,
            "human_reviewed": True,
            "review_method": "test fixture review",
        },
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert main(
        [
            "dataset",
            "--workspace",
            str(tmp_path),
            "validate",
            str(source),
            "--json",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["records"] == 1
