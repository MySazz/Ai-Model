"""Calibrate a pinned NLI cross-encoder on authored Hybrid Agent judgments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


INPUT = Path("/content/semantic-judge-calibration-v1.jsonl")
RUN_DIR = Path("/content/semantic-judge-calibration-v1")
RESULT = Path("/content/semantic-judge-calibration-v1-results.zip")
MODEL_ID = "cross-encoder/nli-deberta-v3-small"
MODEL_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = [json.loads(line) for line in INPUT.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device_map="auto"
    )
    label_ids = {str(label).casefold(): int(index) for index, label in model.config.id2label.items()}
    entailment_id = next(
        (index for label, index in label_ids.items() if "entail" in label), 1
    )
    features = tokenizer(
        [row["premise"] for row in rows], [row["hypothesis"] for row in rows],
        padding=True, truncation=True, return_tensors="pt",
    ).to(model.device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(**features).logits.softmax(dim=-1)[:, entailment_id].cpu().tolist()
    scored = [{**row, "entailment_probability": score} for row, score in zip(rows, probabilities)]
    candidates = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        correct = sum((row["entailment_probability"] >= threshold) == row["expected_entailment"] for row in scored)
        false_accepts = sum(
            row["safety_critical"] and not row["expected_entailment"]
            and row["entailment_probability"] >= threshold for row in scored
        )
        candidates.append({"threshold": threshold, "accuracy": correct / len(scored),
                           "correct": correct, "false_accepts": false_accepts})
    eligible = [candidate for candidate in candidates if candidate["false_accepts"] == 0]
    selected = max(eligible, key=lambda item: (item["accuracy"], item["threshold"])) if eligible else None
    passed = bool(selected and selected["accuracy"] >= 0.9)
    summary = {"schema_version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "records": len(rows), "candidates": candidates, "selected": selected,
               "passed": passed, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "scored.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({"phase": "complete", "result": summary, "archive": str(RESULT)}), flush=True)


if __name__ == "__main__":
    main()
