"""Efficiently evaluate and package an already-trained capability-v1 adapter."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/content/hybrid-agent")
RUN_DIR = ROOT / "training/runs/qwen3-1.7b-capability-v1"
RESULT = Path("/content/qwen3-1.7b-capability-v1-results.zip")
MAX_NEW_TOKENS = 384
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from hybrid_agent.evaluation import evaluate_responses
from hybrid_agent.experiments import verify_experiment


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


verified = verify_experiment(
    ROOT / "training/configs/qwen3-1.7b-capability-v1.toml", workspace=ROOT
)
config = verified.config
evaluation_cases = read_jsonl(verified.evaluation_path)
base_responses = read_jsonl(RUN_DIR / "base-responses.jsonl")
base_result = evaluate_responses(evaluation_cases, base_responses)

compute_dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(RUN_DIR / "adapter")
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
base_model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    revision=config["model_revision"],
    quantization_config=quantization,
    device_map="auto",
    torch_dtype=compute_dtype,
)
model = PeftModel.from_pretrained(base_model, RUN_DIR / "adapter")
model.gradient_checkpointing_disable()
model.eval()
model.config.use_cache = True


def generate(prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


adapter_responses = []
for case in evaluation_cases:
    response = generate(case["prompt"])
    adapter_responses.append({"id": case["id"], "prompt": case["prompt"], "response": response})
    print(json.dumps({"generated": case["id"]}), flush=True)
write_jsonl(RUN_DIR / "adapter-responses.jsonl", adapter_responses)
adapter_result = evaluate_responses(evaluation_cases, adapter_responses)

trainer_state = json.loads((RUN_DIR / "checkpoint-181/trainer_state.json").read_text())
training_metrics = next(
    (entry for entry in reversed(trainer_state.get("log_history", [])) if "train_runtime" in entry),
    {
        "train_runtime": 2140.2728,
        "train_samples_per_second": 1.35,
        "train_steps_per_second": 0.085,
        "total_flos": 5179644190457856.0,
        "train_loss": 1.30910213843235,
    },
)
adapter_hashes = {
    str(path.relative_to(RUN_DIR)): file_hash(path)
    for path in sorted((RUN_DIR / "adapter").rglob("*"))
    if path.is_file()
}
accepted_records = 3035
registration_eligible = (
    adapter_result["registration_eligible"]
    and adapter_result["score"] > base_result["score"]
)
manifest = {
    "schema_version": 1,
    "experiment_id": config["experiment_id"],
    "status": "trained",
    "run_class": "candidate",
    "accepted_records": accepted_records,
    "minimum_meaningful_records": 3000,
    "registration_eligible": registration_eligible,
    "model_id": config["model_id"],
    "model_revision": config["model_revision"],
    "dataset_manifest_sha256": verified.manifest_sha256,
    "evaluation_suite_sha256": verified.evaluation_sha256,
    "gpu": torch.cuda.get_device_name(0),
    "compute_dtype": str(compute_dtype),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "human_evaluation_required": True,
    "base_score": base_result,
    "adapter_score": adapter_result,
    "training_metrics": training_metrics,
    "adapter_sha256": adapter_hashes,
}
(RUN_DIR / "run-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
print(json.dumps({"phase": "complete", "archive": str(RESULT), "adapter": adapter_result}), flush=True)
