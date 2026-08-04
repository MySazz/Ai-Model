"""Small QLoRA diagnostic with explicit assistant-only label masking."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


BUNDLE = Path("/content/hybrid-agent-colab-input.zip")
ROOT = Path("/content/hybrid-agent")
RUN_DIR = ROOT / "training/runs/assistant-mask-diagnostic-v1"
RESULT = Path("/content/assistant-mask-diagnostic-v1-results.zip")
MANIFEST_SHA256 = "ce2eafcfe3a3a6c0a581d75b5a04da499e4b1268fda0f9d14f95d68a8a4a73e0"
DATASET_NAME = "assistant-mask-diagnostic-v1"
EVALUATION_SUITE_NAME = "automated-development-v1"
EVALUATION_EXECUTABLE = False
SECONDARY_EVALUATION_SUITE_NAME = ""
EPOCHS = 5
LEARNING_RATE = 0.0002
REQUIRED_RECALL = 0.45
REQUIRED_IMPROVEMENT = 0.20
REQUIRED_TRANSFER_SCORE = 0.0
MAX_CRITICAL_FAILURES: int | None = None
REQUIRED_HELDOUT_RECALL: float | None = None
REQUIRED_HELDOUT_IMPROVEMENT: float | None = None
SYSTEM_PROMPT = ""
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recall(target: str, response: str) -> float:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "it", "is", "that", "with", "for", "be"}
    wanted = {word for word in re.findall(r"[a-z0-9_]+", target.casefold()) if word not in stop}
    observed = set(re.findall(r"[a-z0-9_]+", response.casefold()))
    return len(wanted & observed) / len(wanted) if wanted else 0.0


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        archive.extractall(ROOT)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "training/requirements-qlora-v1.txt"], check=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from hybrid_agent.evaluation import evaluate_responses

    manifest_path = ROOT / f"datasets/processed/{DATASET_NAME}/manifest.json"
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Diagnostic manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    split_dir = manifest_path.parent
    for split, expected in manifest["output_sha256"].items():
        if sha256(split_dir / f"{split}.jsonl") != expected:
            raise RuntimeError(f"Diagnostic {split} hash mismatch")
    train_records = read_jsonl(split_dir / "train.jsonl")
    validation_records = read_jsonl(split_dir / "validation.jsonl")
    test_records = read_jsonl(split_dir / "test.jsonl")
    evaluation_cases = read_jsonl(
        ROOT / f"datasets/evaluations/{EVALUATION_SUITE_NAME}.jsonl"
    )
    probe_indexes = [round(i * (len(train_records) - 1) / 11) for i in range(12)]
    probes = [train_records[index] for index in probe_indexes]

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device_map="auto", torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
    )

    def generate(prompt: str) -> str:
        messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else [])
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(messages,
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                    pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def probe_rows(records: list[dict]) -> list[dict]:
        rows = []
        for record in records:
            target = record["messages"][-1]["content"]
            response = generate(record["messages"][-2]["content"])
            rows.append({"id": record["id"], "target": target, "response": response,
                         "target_token_recall": recall(target, response)})
        return rows

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    base_probes = probe_rows(probes)
    write_jsonl(RUN_DIR / "base-probes.jsonl", base_probes)
    base_heldout = probe_rows(test_records) if REQUIRED_HELDOUT_RECALL is not None else []
    if base_heldout:
        write_jsonl(RUN_DIR / "base-heldout-probes.jsonl", base_heldout)

    def encode(record: dict) -> dict:
        messages = list(record["messages"])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = full["input_ids"]
        if input_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"Chat-template prefix mismatch for {record['id']}")
        if len(prompt_ids) >= len(input_ids):
            raise RuntimeError(f"No assistant labels for {record['id']}")
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}

    train_data = Dataset.from_list([encode(row) for row in train_records])
    validation_data = Dataset.from_list([encode(row) for row in validation_records])
    supervised = sum(sum(token != -100 for token in row["labels"]) for row in train_data)
    masked = sum(sum(token == -100 for token in row["labels"]) for row in train_data)
    if not supervised or not masked:
        raise RuntimeError("Assistant-mask invariant failed")
    print(json.dumps({"phase": "mask-verified", "records": len(train_data),
                      "supervised_tokens": supervised, "masked_tokens": masked}), flush=True)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    from peft import get_peft_model
    model = get_peft_model(model, lora)
    args = TrainingArguments(output_dir=str(RUN_DIR / "checkpoints"), seed=20260803,
        num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine", warmup_ratio=0.05,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=True, fp16=True, optim="paged_adamw_8bit", logging_steps=5,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    result = trainer.train()
    trainer.save_model(RUN_DIR / "adapter")
    tokenizer.save_pretrained(RUN_DIR / "adapter")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    adapter_probes = probe_rows(probes)
    write_jsonl(RUN_DIR / "adapter-probes.jsonl", adapter_probes)
    adapter_heldout = probe_rows(test_records) if REQUIRED_HELDOUT_RECALL is not None else []
    if adapter_heldout:
        write_jsonl(RUN_DIR / "adapter-heldout-probes.jsonl", adapter_heldout)
    responses = [{"id": case["id"], "prompt": case["prompt"], "response": generate(case["prompt"])}
                 for case in evaluation_cases]
    write_jsonl(RUN_DIR / "transfer-responses.jsonl", responses)
    if EVALUATION_EXECUTABLE:
        from hybrid_agent.executable_evaluation import run_python_checks
        transfer = evaluate_responses(
            evaluation_cases, responses, executable_judge=run_python_checks
        )
    else:
        transfer = evaluate_responses(evaluation_cases, responses)
    secondary_transfer = None
    if SECONDARY_EVALUATION_SUITE_NAME:
        secondary_cases = read_jsonl(
            ROOT / f"datasets/evaluations/{SECONDARY_EVALUATION_SUITE_NAME}.jsonl"
        )
        secondary_responses = [
            {"id": case["id"], "prompt": case["prompt"], "response": generate(case["prompt"])}
            for case in secondary_cases
        ]
        write_jsonl(RUN_DIR / "secondary-transfer-responses.jsonl", secondary_responses)
        secondary_transfer = evaluate_responses(secondary_cases, secondary_responses)
    base_recall = sum(row["target_token_recall"] for row in base_probes) / len(base_probes)
    adapter_recall = sum(row["target_token_recall"] for row in adapter_probes) / len(adapter_probes)
    base_heldout_recall = (
        sum(row["target_token_recall"] for row in base_heldout) / len(base_heldout)
        if base_heldout else None
    )
    adapter_heldout_recall = (
        sum(row["target_token_recall"] for row in adapter_heldout) / len(adapter_heldout)
        if adapter_heldout else None
    )
    heldout_passed = (
        REQUIRED_HELDOUT_RECALL is None
        or (
            adapter_heldout_recall is not None
            and base_heldout_recall is not None
            and adapter_heldout_recall >= REQUIRED_HELDOUT_RECALL
            and adapter_heldout_recall
            >= base_heldout_recall + (REQUIRED_HELDOUT_IMPROVEMENT or 0.0)
        )
    )
    transfer_passed = (transfer["score"] >= REQUIRED_TRANSFER_SCORE and
                       (MAX_CRITICAL_FAILURES is None or
                        len(transfer["critical_failures"]) <= MAX_CRITICAL_FAILURES))
    secondary_passed = (
        secondary_transfer is None
        or (secondary_transfer["score"] >= 10 / 12 and not secondary_transfer["critical_failures"])
    )
    diagnostic_passed = (adapter_recall >= REQUIRED_RECALL and
                         adapter_recall >= base_recall + REQUIRED_IMPROVEMENT and
                         heldout_passed and transfer_passed and secondary_passed)
    summary = {"schema_version": 1, "status": "passed" if diagnostic_passed else "failed",
               "evaluation_suite": EVALUATION_SUITE_NAME,
               "assistant_mask_verified": True, "train_records": len(train_records),
               "validation_records": len(validation_records), "base_probe_recall": base_recall,
               "adapter_probe_recall": adapter_recall, "required_recall": REQUIRED_RECALL,
               "base_heldout_recall": base_heldout_recall,
               "adapter_heldout_recall": adapter_heldout_recall,
               "required_heldout_recall": REQUIRED_HELDOUT_RECALL,
               "required_heldout_improvement": REQUIRED_HELDOUT_IMPROVEMENT,
               "heldout_passed": heldout_passed,
               "required_improvement": REQUIRED_IMPROVEMENT, "training_metrics": result.metrics,
               "required_transfer_score": REQUIRED_TRANSFER_SCORE,
               "max_critical_failures": MAX_CRITICAL_FAILURES,
               "transfer_passed": transfer_passed,
               "transfer_evaluation": transfer,
               "secondary_evaluation_suite": SECONDARY_EVALUATION_SUITE_NAME or None,
               "secondary_transfer_passed": secondary_passed,
               "secondary_transfer_evaluation": secondary_transfer,
               "gpu": torch.cuda.get_device_name(0)}
    (RUN_DIR / "diagnostic-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({"phase": "complete", "result": summary, "archive": str(RESULT)}), flush=True)


if __name__ == "__main__":
    main()
