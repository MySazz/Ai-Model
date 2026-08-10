# Meaningful capability training and automated evaluation plan

## Objective

Produce a useful Qwen3-1.7B adapter for the Hybrid Agent's coding, tool-use,
research, infrastructure, privacy, and general-assistance behaviors. Training
and evaluation remain separate: training examples teach behavior, while unseen
evaluation cases measure whether it generalized.

The current 12-record batch validates the pipeline only. The first meaningful
run targets 3,000–5,000 reviewed training conversations with a separate,
immutable automated evaluation suite. Dataset size alone is not a capability
claim; correctness, coverage, diversity, and executable verification matter
more than raw volume.

## Training curriculum

| Capability | Target records | Share | Required example types |
| --- | ---: | ---: | --- |
| Coding | 1,200–1,800 | 35% | implementation, debugging, tests, review, secure file handling, concurrency |
| Tool use | 650–1,000 | 20% | inspect-plan-act loops, tool-result grounding, failures, retries, approvals |
| Research | 400–700 | 14% | source comparison, uncertainty, citations, conflicting and unavailable evidence |
| Infrastructure | 350–600 | 12% | deploy diagnosis, rollout, rollback, migrations, monitoring, incident response |
| Privacy and safety | 300–500 | 10% | secrets, disclosure gates, prohibited actions, prompt injection, redaction |
| General assistance | 250–450 | 9% | planning, concise rewriting, explanation, clarification, calibrated uncertainty |

At least 20% of coding records should contain executable tests. At least 30% of
tool-use records should include a failed or inconclusive tool result. At least
15% of all records should be contrastive examples that demonstrate a plausible
wrong approach followed by the correct bounded behavior without teaching the
model to emit hidden reasoning.

## Record quality requirements

Every training record must:

1. Have approved provenance and license metadata.
2. Be technically checked by an executable validator where possible.
3. Contain no credentials, private data, or identifying PII.
4. Avoid copying prompts, solutions, fixtures, or distinctive wording from any
   evaluation suite.
5. Teach an observable response or tool behavior, not claims about being safe
   or capable.
6. Include realistic variation in phrasing, constraints, languages, repository
   layouts, failure modes, and task difficulty.
7. Pass duplicate and near-duplicate screening before split preparation.

Synthetic records are acceptable only when their answers are independently
validated. Automatically generated code examples must execute in an isolated,
network-disabled test environment before acceptance. A human should review a
statistically meaningful sample from each generation batch even when every
record has automated checks.

## Automated held-out evaluation suite

Create at least 200 unseen cases for development and 300 unseen cases for the
final acceptance suite. Freeze the final suite before the first meaningful
training run and keep its prompts, fixtures, and expected results out of all
training-generation context.

| Evaluator | Cases | Automatic pass conditions |
| --- | ---: | --- |
| Python coding | 100 | Generated solution passes hidden unit and property tests within time and resource limits |
| Code repair | 50 | Patch applies cleanly and changes failing tests to passing without regressions |
| Tool policy | 75 | Exact allowed/approval/prohibited decision and valid structured tool arguments |
| Tool grounding | 50 | Final answer agrees with supplied tool evidence and does not invent observations |
| Research grounding | 50 | Claims map to supplied source IDs; unsupported or conflicting facts are qualified |
| Infrastructure | 50 | Required rollout, monitoring, abort, and recovery elements match structured rubric |
| Privacy and safety | 50 | Secrets are never reproduced; prohibited actions are refused; safe alternatives remain useful |
| General retention | 75 | Format, constraint, planning, rewriting, and instruction-following assertions pass |

Use deterministic decoding for comparisons. Record both overall and
capability-level scores. Safety-critical cases use a hard gate: one secret
disclosure or prohibited action prevents adapter registration regardless of the
aggregate score.

## Evaluation implementation

Each case should use a machine-readable record containing:

- stable ID and capability;
- prompt and optional workspace fixture;
- evaluator type and evaluator version;
- maximum runtime and output size;
- assertions or hidden test reference;
- critical-failure classification;
- source and license metadata;
- SHA-256 hashes for every fixture and evaluator.

Run model generations in one process and evaluators in an isolated subprocess
with no network, no inherited secrets, a temporary workspace, explicit resource
limits, and captured stdout/stderr. Never execute generated infrastructure or
shell actions against a real service. Infrastructure tasks are scored against
structured plans or simulated tool APIs.

## Automatic training gate

The Colab training workflow may remain one-click, but it should refuse to train
unless all of the following are true:

- at least 3,000 accepted training records exist;
- every record passes schema, provenance, license, privacy, duplicate, and
  contamination checks;
- all processed split and evaluator hashes match the frozen manifest;
- the base model has an automated score on the same final suite;
- the run records model revision, seed, packages, GPU, dataset hashes, metrics,
  and adapter hashes.

Register the adapter only when it improves the primary coding and tool-use
score, has no critical safety failures, and does not regress general capability
by more than five percentage points. A calibration run on fewer records can
verify mechanics but must be labeled `pipeline-only` rather than `candidate`.

## Build sequence

1. Define evaluator schemas and a local sandboxed evaluation runner.
2. Author 25–50 representative cases per evaluator and verify that the pinned
   base model produces a useful score distribution.
3. Expand to the frozen 300-case acceptance suite without exposing it to the
   training-data generator.
4. Generate training data in batches of 100–250 records, validate each batch,
   and retain record-level provenance and review decisions.
5. Train checkpoints at approximately 500, 1,500, 3,000, and 5,000 accepted
   records. Stop expanding when held-out gains flatten or regress.
6. Compare every adapter with the pinned base model using identical decoding,
   evaluator versions, and hardware-independent pass criteria.
