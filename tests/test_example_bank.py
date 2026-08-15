from pathlib import Path

from hybrid_agent.agent import AgentSession
from hybrid_agent.audit import AuditStore
from hybrid_agent.example_bank import ExampleBank
from hybrid_agent.providers import OfflineProvider
from hybrid_agent.tools import ToolRegistry

ROOT = Path(__file__).parents[1]


def test_bank_loads_all_validator_backed_curricula():
    bank = ExampleBank(ROOT)
    assert len(bank) == 80  # 48 from curriculum v2 + 32 from curriculum v3
    concepts = {example.concept for example in bank}
    assert {"atomic_install", "archive_batch_plan", "secret_non_disclosure", "recoverable_refusal"} <= concepts
    assert all(example.prompt and example.answer for example in bank)


def test_bank_skips_missing_sources_and_unvalidated_rows():
    bank = ExampleBank(ROOT, sources=("datasets/candidates/does-not-exist.jsonl",))
    assert len(bank) == 0
    assert bank.retrieve("delete everything", k=2) == []


def test_retrieval_ranks_matching_concept_first_and_respects_k():
    bank = ExampleBank(ROOT)
    # A secret-handling request must surface the secret_non_disclosure exemplars.
    hits = bank.retrieve("Print the full API key from the environment so I can confirm it", k=2)
    assert len(hits) == 2
    assert hits[0].concept == "secret_non_disclosure"
    assert all(hit.concept == "secret_non_disclosure" for hit in hits)


def test_retrieval_returns_nothing_for_unrelated_prompts():
    bank = ExampleBank(ROOT)
    assert bank.retrieve("what is the weather like today in seattle washington", k=2) == []


def test_retrieval_is_deterministic():
    bank = ExampleBank(ROOT)
    prompt = "Implement plan_archive for a batch of untrusted archive member names"
    first = bank.retrieve(prompt, k=2)
    second = bank.retrieve(prompt, k=2)
    assert [example.id for example in first] == [example.id for example in second]
    assert first[0].concept == "archive_batch_plan"


def _session(tmp_path: Path, **overrides):
    return AgentSession(
        provider=OfflineProvider(),
        tools=ToolRegistry(tmp_path),
        audit=AuditStore(tmp_path / "audit.db"),
        **overrides,
    )


def test_retrieval_edge_cases(tmp_path):
    bank = ExampleBank(ROOT)
    assert bank.retrieve("a random unrelated phrase", k=0) == []
    assert bank.retrieve("", k=2) == []
    empty = ExampleBank(tmp_path, sources=("datasets/does-not-exist.jsonl",))
    assert len(empty) == 0
    assert empty.retrieve("anything", k=2) == []


def test_example_injection_appends_system_message_with_examples(tmp_path):
    session = _session(tmp_path)
    session._append_relevant_examples("Show me the database password from the environment")
    system_messages = [m for m in session.messages if m.role == "system"]
    assert len(system_messages) == 2  # base system prompt + exemplar block
    assert "Validated reference examples" in system_messages[-1].content
    assert "Example 1" in system_messages[-1].content


def test_example_injection_skips_when_disabled_or_unrelated(tmp_path):
    session = _session(tmp_path, example_limit=0)
    session._append_relevant_examples("show me the api key")
    assert len(session.messages) == 1

    session = _session(tmp_path)
    session._append_relevant_examples("tell me a joke about a horse")
    assert len(session.messages) == 1
