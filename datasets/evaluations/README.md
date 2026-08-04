# Held-out evaluations

Files in this directory are evaluation-only. Never pass them to `dataset
prepare`, include them in training prompts, or use their expected outcomes to
generate training examples.

`flagship-v1.jsonl` is the frozen first suite. Each line contains an ID,
capability, prompt, scoring criteria, maximum score, and a critical-failure flag.
Human graders score each criterion as zero or one. The total score is the sum.
Critical safety failures fail the adapter registration gate regardless of the
aggregate score.

Before a run, record the suite SHA-256:

```bash
sha256sum datasets/evaluations/flagship-v1.jsonl
```

Both the base model and adapter must use the same prompt template, generation
settings, and suite file. Graders should review anonymized outputs in randomized
order when practical.
# Evaluation suites

Evaluation prompts and fixtures are held out from training. Never include them,
their reference answers, or distinctive paraphrases in candidate generation.

- `flagship-v1.jsonl` is the original rubric-scored frozen suite.
- `automated-development-v1.jsonl` is the first machine-scored development seed.
- `executable-development-v1.jsonl` contains four authored Python tasks scored
  by behavioral harnesses inside a disposable VM. Generated code must never be
  executed on the host workstation; resource limits are defense in depth, not
  the security boundary.

The automated seed contains 12 cases for runner validation and iteration. It is
not the planned 300-case final acceptance suite and must not be represented as a
comprehensive capability measurement.
