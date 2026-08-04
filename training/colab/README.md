# Google Colab QLoRA run

Use `qwen3-1.7b-qlora-v1.ipynb` with a Colab GPU runtime. An L4, A10G, or A100
is preferred; a T4 should fit this 1.7B QLoRA experiment but will use FP16.

## Before opening Colab

If this repository is on GitHub, copy its clone URL and enter it in the
notebook's parameter cell. Otherwise create the small input bundle locally:

```bash
bash scripts/create_colab_bundle.sh
```

Upload both the notebook and the resulting `hybrid-agent-colab-input.zip` to
Colab when prompted. The bundle contains source data and manifests, not model
weights or secrets.

## In Colab

1. Select **Runtime → Change runtime type → GPU**.
2. Choose **Runtime → Run all**.
3. When prompted, upload `hybrid-agent-colab-input.zip`.
4. Leave the tab open while setup, model download, and training run automatically.
5. Save the automatically downloaded `qwen3-1.7b-qlora-v1-results.zip`.

The capability-v2 easy mode verifies 3,500 accepted records, trains immediately, and
automatically scores the pinned base model and adapter. It requires no answer
review or manual scoring and exports responses, scores, training metrics,
environment metadata, and adapter hashes.

Expect substantially longer than the 12-record calibration run. The capability
configuration trains 3,138 records for one epoch and evaluates on 173 validation
records. Keep the Colab tab and runtime connected until the results ZIP downloads.

For CLI execution, upload `hybrid-agent-colab-input.zip` to `/content`, upload
`run_capability_v2.py`, and execute the runner. It produces
`/content/qwen3-1.7b-capability-v2-results.zip` without manual answer grading.

Training the current 12-record calibration batch proves the workflow but is too
small to establish a useful capability improvement. Do not register the adapter
from this batch as a production model.

## Family-held-out transfer diagnostic v3

Before another candidate-scale run, use `run_transfer_diagnostic_v3.py`. Its
48-record curriculum contains four complete, independently authored paraphrase
rounds spanning all 12 behavioral concepts. Group-aware preparation assigns two
entire rounds to training, one to validation, and one to test; no round crosses
a split.

Upload `hybrid-agent-colab-input.zip`,
`run_assistant_mask_diagnostic_v1.py`, and `run_transfer_diagnostic_v3.py`, then
run:

```bash
python run_transfer_diagnostic_v3.py
```

The runner exports `/content/transfer-diagnostic-v3-results.zip`. It requires
assistant-only masking, improved held-in recall, improved recall on the fully
held-out paraphrase round, at least `6/12` automated development cases, and zero
critical failures. This remains a diagnostic and is never registration-eligible.
