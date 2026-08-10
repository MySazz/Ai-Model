# Datasets

Training data stays separated by lifecycle:

- `raw/`: immutable source exports after licensing and privacy approval
- `candidates/`: proposed records that are blocked from training until human review
- `examples/`: small, safe records committed for documentation and tests
- `processed/`: generated normalized train/validation/test files
- `manifests/`: source, license, ownership, and collection documentation
- `evaluations/`: held-out capability tests that must never enter training
- `generation/`: training-only generation tasks, raw candidate batches, and
  filter-calibration fixtures that must remain distinct from evaluations
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

`metadata.reviewed` means that the declared `review_method` completed; it does
not imply a human review. `metadata.human_reviewed` records that separately.
Human reviewers must check correctness, safety, provenance, license, clarity,
and overlap with held-out evaluations before setting `human_reviewed: true`.
Generated records may pass an automated quality gate while remaining explicitly
marked `human_reviewed: false`.

Generated candidates use `training/schemas/generated-candidate.schema.json` and
are filtered with `scripts/filter_execution_guided_candidates.py`. Authored
filter fixtures carry `calibration_only: true`; they can exercise the pipeline
but can never open its training-ready gate. Filtering also requires the operator
to supply an approved output license and a written license basis; the pipeline
does not infer ownership from model output.
