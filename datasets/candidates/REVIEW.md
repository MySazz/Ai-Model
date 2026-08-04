# Candidate review procedure

Review `coding-tool-v1-batch-001.jsonl` one record at a time. For every record:

1. Confirm the answer is technically correct and complete for the prompt.
2. Confirm it teaches bounded inspection, honest reporting, and appropriate
   approval behavior rather than merely mentioning those values.
3. Confirm the example contains no credential, private data, or identifying PII.
4. Confirm `metadata.source`, `metadata.license`, and `metadata.category` are
   accurate.
5. Compare it with `datasets/evaluations/flagship-v1.jsonl`; reject or rewrite
   anything that copies a held-out prompt or expected answer.
6. Record requested corrections by ID. Change `reviewed` to `true` only after a
   human accepts the final text.

Batch 001 contains 12 records: eight coding explanations, two coding tool-call
traces, one infrastructure approval trace, and one privacy behavior example.
Its completed record-level decision log is in
`coding-tool-v1-batch-001.review.md`. It is an initial quality-calibration batch,
not yet enough data for meaningful training.
