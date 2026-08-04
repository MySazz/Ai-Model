# Project Checkpoint

**Updated:** August 3, 2026
**Resume from:** Author at least eight independent families per behavior before the next GPU run

## Project charter

Build a local-first, privacy-conscious, hybrid general AI system for Jon and
Johnnie. Its flagship capabilities are coding, research, vision, and
infrastructure management, while retaining strong general reasoning, writing,
planning, and assistance abilities.

The project is CLI-first. It uses graduated autonomy: low-risk inspection may
run automatically, consequential actions require approval, and prohibited
actions cannot be authorized by the model itself.

## Decisions locked

- The entire reproducible project lives in this repository directory.
- Python owns the initial agent, model, training, and evaluation layers.
- Rust or other languages enter only for a measured responsibility.
- The system is hybrid: local models for privacy and routine work, cloud models
  for harder tasks after disclosure checks and approval.
- Model providers remain swappable behind one interface.
- Training data is validated before GPU training is introduced.
- Documents and current facts belong in retrieval; behavioral examples belong
  in fine-tuning datasets.

## Completed milestones

1. Headless agent core and CLI
2. Provider-neutral model contract
3. Offline and Ollama providers
4. Workspace-scoped tools and graduated permission policy
5. SQLite audit history and explicit structured memory
6. Privacy classification and cloud-disclosure gates
7. Safe, shell-free command execution with dynamic risk levels
8. Durable research sources with hashes and line provenance
9. Validated PNG, JPEG, and GIF inputs with Ollama vision transport
10. Exact-match reversible editing with diff approval and backups
11. Conversational/tool-calling training-data validation
12. License, provenance, review, secret, PII, and duplicate data gates
13. Deterministic train/validation/test preparation with hash manifests
14. Qwen3-1.7B selected and pinned at an immutable upstream revision
15. First QLoRA configuration and dependency lock recorded
16. Frozen 11-case held-out evaluation suite defined before training
17. Twelve-record coding/tool-use calibration batch reviewed and accepted
18. License, provenance, privacy, duplicate, and evaluation-contamination audit recorded
19. Deterministic `coding-tool-v1` splits and manifest frozen by SHA-256
20. Experiment integrity entry point verifies source, split, manifest, and evaluation hashes
21. Rubric-aware optional scoring validates per-case ranges and flags critical failures
22. Colab easy mode trains and exports automatically without requiring human scoring
23. Meaningful-capability curriculum and 300-case automated acceptance-suite plan defined
24. First Colab QLoRA calibration run completed on a Tesla T4 and artifact hashes verified
25. Machine-scored evaluation runner, case schema, and 12-case development seed added
26. Curriculum manifest freezes a 3,000-record minimum and 4,000-record target
27. Colab labels undersized datasets `pipeline-only` and prevents registration claims
28. Pinned Apache-2.0 coding, tool-use, and DevOps sources imported reproducibly
29. Capability-v1 corpus passes validation with 3,200 records, zero errors, and zero warnings
30. Frozen splits contain 2,890 training, 145 validation, and 165 test records
31. Colab candidate workflow automatically compares base and adapter responses
32. Capability-v1 candidate trained on a T4 and was rejected after automated regression
33. Capability-v2 added 900 safety/reliability examples and passed strict dataset validation
34. Capability-v2 trained on a T4 and was rejected after reproducing a 0/12 adapter score
35. Assistant-only masking diagnostic verified 4,582 masked prompt tokens and 8,449 supervised assistant tokens
36. Diagnostic improved held-in recall by 0.235 and held-out transfer from 0/12 to 3/12
37. Concise assistant-mask diagnostic v2 achieved perfect 1.000 held-in recall
38. Combined generalization diagnostic rejected after templated data reduced transfer to 0/12
39. Independent paraphrase diagnostic reached the best adapter transfer score: 5/12
40. System-policy variant improved direct safety wording but reduced total transfer to 1/12
41. Family-held-out split implemented and verified across four complete paraphrase families
42. Two-family training proved insufficient, scoring 4/12 with a secret-handling regression
41. Group-aware dataset preparation now prevents paraphrase-family leakage
42. Transfer diagnostic v3 locally frozen with complete 24/12/12 family-held-out splits

## Current verification state

- 58 automated tests pass.
- Python source and tests compile.
- The example SFT dataset validates with zero errors and warnings.
- The example dataset prepares into deterministic, hash-verified splits.
- No cloud credentials have been added.
- The first pipeline-calibration adapter trained successfully for two optimizer
  steps on the 12-record batch. Its verified compact result is recorded in
  `training/results/qwen3-1.7b-qlora-v1-calibration-run.json`; the run is not a
  capability claim and is not eligible for registration.
- The pinned base-model baseline was attempted but could not run: the model and ML
  dependencies are absent and the NVIDIA driver is unavailable. The evidence-backed
  blocked result is recorded in `training/results/qwen3-1.7b-base-baseline.json`.
- The local GTX 1650 is detected, but `nvidia-smi` currently cannot communicate
  with the driver; the reference QLoRA run therefore targets a cloud GPU.
- Capability-v1 completed one epoch in 2,140 seconds with train loss 1.3091 and
  validation loss improving to 0.8662, but behavioral evaluation regressed from
  1/12 base cases to 0/12 adapter cases and added a critical safety failure. The
  adapter is rejected and must not be registered or deployed. The verified result
  is `training/results/qwen3-1.7b-capability-v1.json`.
- Capability-v2 completed one epoch in 1,757 seconds with train loss 2.7087.
  Validation loss improved from 3.2729 to 2.2262, but behavioral evaluation
  again scored 0/12 with five critical failures. The adapter is rejected. Its
  compact result is `training/results/qwen3-1.7b-capability-v2.json` and its
  verified archive remains under ignored `training/runs/`.
- The 101-record assistant-mask diagnostic trained in 234 seconds. Held-in target
  recall improved from 0.173 to 0.408, narrowly missing the deliberately strict
  0.450 threshold, while held-out evaluation improved to 3/12. It remains unsafe
  and is not deployable, but it establishes that explicit assistant-only labels
  materially improve behavioral transfer. Result:
  `training/results/assistant-mask-diagnostic-v1.json`.
- Diagnostic v2 used 105 concise, concept-balanced records and trained in 302
  seconds. It passed the training-mechanics criterion: held-in recall increased
  from 0.253 to 1.000. Held-out transfer remained weak at 2/12 with five critical
  failures, proving the remaining blocker is paraphrase/generalization coverage,
  not label masking. Result: `training/results/assistant-mask-diagnostic-v2.json`.
- Generalization diagnostic v1 combined 900 verbose generated examples with 133
  concise examples and used assistant-only masking. Held-in recall reached 0.740,
  but transfer fell to 0/12 because the verbose template signal dominated and
  produced repetitive, misplaced safety language. The mixture is rejected;
  result: `training/results/generalization-diagnostic-v1.json`.
- Transfer diagnostic v1 removed verbose templates and added 48 independently
  authored paraphrases. It reached the best result so far: 5/12, with research
  at 2/2 and three critical failures. It is not deployable. Result:
  `training/results/transfer-diagnostic-v1.json`.
- Transfer diagnostic v2 added a production behavioral system policy and used
  four epochs. Its secret response was safe, but broad transfer fell to 1/12;
  the variant is rejected. Result: `training/results/transfer-diagnostic-v2.json`.
- Transfer diagnostic v3 grouped paraphrases before splitting: 24 train, 12
  validation, and 12 test, with every behavior represented in each partition.
  Validation-family loss bottomed near epoch 15 and then rose, while external
  transfer reached 4/12 and secret handling regressed. This proves two training
  phrasings per behavior are inadequate. Result:
  `training/results/transfer-diagnostic-v3.json`.
- Transfer diagnostic v4 expanded the curriculum to eight independently authored
  families (72 train, 12 validation, 12 test). Eight epochs produced only 3/12
  external passes and four critical failures; held-in recall regressed to 0.201.
  Result: `training/results/transfer-diagnostic-v4.json`.
- Transfer diagnostic v5 extended that exact controlled run to 15 epochs. Recall
  improved from 0.223 to 0.507, but unseen-family validation loss bottomed at
  epoch 7 (2.632) and then rose to 3.190 while external transfer remained 4/12
  with four critical failures. More epochs increase memorization, not capability.
  The adapter is rejected. Result: `training/results/transfer-diagnostic-v5.json`.
- The deterministic v2 evaluator replaces brittle literal phrases with named,
  auditable regex concepts and fixes a duplicate JSON key that silently removed
  migration checks. It re-scores transfer-v5 at 5/12 with three genuine critical
  failures. All secret-leak and prohibited-action gates remain deterministic.
- Pinned Qwen3-8B zero-shot scored 1/12 on evaluator v2 with five critical
  failures, including explicit prohibited deletion and token-printing advice.
  Capacity alone is not alignment; this base must not be deployed. Result:
  `training/results/qwen3-8b-baseline-v1.json`.
- Pinned Qwen3-4B-Instruct-2507 under the explicit agent policy established the
  strongest safe base: 7/12 with zero critical failures on the finalized v2
  evaluator. Result: `training/results/qwen3-4b-policy-baseline-v1.json`.
- Qwen3-4B policy transfer v1 used assistant-only QLoRA and restored the best
  family-held-out checkpoint (epoch 6, validation loss 2.123). Finalized-v2
  rescoring reached 9/12, but tuning introduced one critical unsupported-source
  claim and severe repetition in the atomic-write answer. It is rejected despite
  the aggregate gain. Result: `training/results/qwen3-4b-policy-transfer-v1.json`.
- Transfer curriculum v4 contains 120 records in eight complete families: 96
  balanced replay examples plus 24 validator-backed targets for unavailable
  sources, atomic JSON replacement, and trusted rooted paths. The frozen split
  is 90 train, 15 validation, and 15 test with no family leakage.
- Qwen3-4B policy transfer v2 used the targeted curriculum, six epochs, and half
  the prior learning rate. It regressed held-out recall from 0.269 to 0.225 and
  scored only 5/12 after final evaluator calibration. The adapter is rejected.
  Result: `training/results/qwen3-4b-policy-transfer-v2.json`.
- The untuned safe 4B base with two retrieved, validated training examples per
  request reached the development target: 10/12 and zero critical failures.
  It missed controlled rollout and atomic failure cleanup. Because evaluator v2
  was calibrated while inspecting development outputs, this is not independent
  evidence and is not registration-eligible. Result:
  `training/results/qwen3-4b-policy-retrieval-v1.json`.
- The first untouched holdout contains 60 new prompts, five per frozen v2 rubric,
  at SHA-256 `01810d0b7e6bbc213c1da585745fed10e54d7ecee13d11c1c52937a8ad7a7f07`.
  The unchanged retrieval configuration scored 23/60 (38.3%): general 100%,
  coding/tool/research 40%, infrastructure 30%, and safety 10%. It failed the
  independent gate. Manual inspection of all 11 automatically critical responses
  found safe refusals in every case, demonstrating severe regex false negatives;
  the frozen automated score is retained unchanged. Result:
  `training/results/qwen3-4b-policy-holdout-v3.json`.
- Semantic-judge calibration v1 tested the pinned 0.1B DeBERTa NLI cross-encoder
  on 20 separately authored, balanced pairs. Its best zero-safety-false-accept
  threshold scored only 15/20 (75%), below the frozen 90% gate, so the candidate
  is rejected. Result: `training/results/semantic-judge-calibration-v1.json`.
- Semantic-judge calibration v2 tested pinned Phi-4-mini-instruct with three
  paraphrased binary judgments and majority vote. Both the initial and inert-data
  hardened prompts scored 18/20 (90%), but each falsely accepted the same critical
  secret-disclosure case. The candidate is rejected under the zero-false-accept
  rule. Result: `training/results/semantic-judge-calibration-v2.json`.
- The evaluator now supports `executable_assertions`: an external isolated
  runner must return a complete boolean map for declared named checks, while the
  core aggregator validates completeness and applies check-level critical gates.
  No generated code is executed in the local evaluator process.
- Executable development v1 contains four behaviorally scored Python tasks at
  SHA-256 `d40bd49927bfc70ea79a33b4a58736c62a3afc2ab640932fed152e873db4921a`.
  Reference implementations pass all checks and deliberately naive versions
  fail. The frozen Qwen3-4B baseline scored 1/4: structured traces passed, while
  atomic persistence, rooted-path containment, and controlled rollout failed
  critical checks. Result:
  `training/results/qwen3-4b-executable-development-v1.json`.
- Executable curriculum v1 contains 32 executable, validator-backed targets in
  eight complete families. The frozen split is 24 train, 4 validation, and 4
  test with no family leakage; the candidate SHA-256 is
  `30e66afb915ebab5bc808e98076fbf838d781cdc4524f85f2a7276a6a37847e4`.
- Qwen3-4B executable transfer v1 trained for six epochs and reduced validation
  loss to 0.936, but behavioral transfer failed. Executable score remained 1/4,
  training-family recall fell 0.574→0.529, held-out-family recall fell
  0.680→0.644, and policy score collapsed to 2/12 with three critical failures.
  The adapter is rejected. Result:
  `training/results/qwen3-4b-executable-transfer-v1.json`.
- Execution-guided filtering v1 is implemented with eight training-only tasks
  across all six capabilities. It performs executable or deterministic behavior
  checks, critical-failure accounting, repetition rejection, global exact/near
  deduplication, per-task coverage, and capability-balance gating. Its authored
  calibration accepted 8/12, correctly rejected two behavioral failures, one
  duplicate, and one repetitive answer, and remained training-ineligible because
  calibration fixtures can never open the gate. Result:
  `training/results/execution-guided-filter-calibration-v1.json`.
- Qwen3-4B multi-candidate generation v1 sampled four seeded answers for each of
  the eight tasks and filtered all 32 without training. Only 5/32 passed (three
  constructive-feedback and two unavailable-source answers); 27 had behavioral
  failures, four were excessively repetitive, and 21 tripped critical checks.
  Coding, infrastructure, safety, and tool-use coverage were all zero, so the
  corpus gate stayed closed. Result:
  `training/results/qwen3-4b-multicandidate-generation-v1.json`.

## Resume commands

```bash
cd '/home/j-alien/Documents/GitRepositories/Ai Model'
PYTHONPATH=src python3 -m pytest
PYTHONPATH=src python3 -m hybrid_agent.cli doctor
PYTHONPATH=src python3 -m hybrid_agent.cli dataset validate \
  datasets/examples/sft.sample.jsonl
```

## Exact next milestone

Do not rerun capability-v1 or capability-v2 unchanged. Both improved token-level
loss while failing behavioral transfer.

1. Preserve holdout-v3 and its 23/60 score unchanged; it has already been seen.
2. Replace regex-only semantic grading with an automatic two-tier evaluator:
   deterministic hard gates for leaks/actions plus an independently validated
   semantic-entailment or judge layer for paraphrase completeness.
3. Keep the semantic fixture and gate frozen. DeBERTa NLI was rejected at 75%;
   Phi-4-mini was rejected at 90% because it had one critical false accept. Do
   not tune either candidate further on these now-seen records.
4. Executable development, filtering, and the one-shot generation baseline are
   operational. Next add bounded execution-feedback repair: return only named
   failed checks to the generator, allow at most two repairs, and rescore through
   the unchanged filter. Require two diverse passes per task before policy replay.
5. After the evaluator is frozen, author another unseen holdout and require zero
   actual safety failures, at least 80% overall, and every capability at 70%+.
   Do not run another positive-only QLoRA experiment.

Do not start by downloading a large model or spending GPU money. Hardware,
license, dataset quality, and baseline evaluations must be established first.

## Important repository state

This directory is its own Git repository connected to
`https://github.com/Jdrexx/Ai-Model`. Publish validated milestones through scoped
branches and pull requests; do not stage sibling repositories or ignored runtime
artifacts.

Large datasets, model weights, training runs, `.env`, and `.hybrid-agent/`
runtime state are excluded from ordinary Git. Their manifests, hashes,
licenses, configurations, and reproduction instructions belong in Git.
