# Datasets

Training data stays separated by lifecycle:

- `raw/`: immutable source exports after licensing and privacy approval
- `candidates/`: proposed records that are blocked from training until human review
- `examples/`: small, safe records committed for documentation and tests
- `processed/`: generated normalized train/validation/test files
- `manifests/`: source, license, ownership, and collection documentation
- `evaluations/`: held-out capability tests that must never enter training
- `licenses/`: license texts or usage approvals for incorporated datasets

Large datasets must not be committed to ordinary Git. Commit their manifest,
hash, license, preparation configuration, and a small non-sensitive sample.

The planned scale, curriculum, automated evaluator mix, and registration gates
for the first meaningful adapter are defined in
`docs/training-capability-plan.md`.

Every JSONL record must conform to `training/schemas/sft-record.schema.json`.
Validation also performs checks that JSON Schema alone cannot express, including
secret screening, possible PII detection, duplicate checks, conversation
quality checks, and approved-license enforcement.

Candidate records deliberately use `"reviewed": false`. Reviewers must check
correctness, safety, provenance, license, clarity, and overlap with held-out
evaluations before changing that field. The validator's resulting `reviewed`
errors are an intentional workflow gate, not a reason to weaken validation.
