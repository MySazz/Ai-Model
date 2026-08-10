"""Non-interactive Colab CLI runner for the capability-v1 QLoRA experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


BUNDLE = Path("/content/hybrid-agent-colab-input.zip")
ROOT = Path("/content/hybrid-agent")
RESULT_ARCHIVE = Path("/content/qwen3-1.7b-capability-v1-results.zip")
CONFIG_PATH = Path("training/configs/qwen3-1.7b-capability-v1.toml")
MINIMUM_MEANINGFUL_RECORDS = 3000
MAX_NEW_TOKENS = 384


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not BUNDLE.is_file():
        raise RuntimeError(f"Missing uploaded bundle: {BUNDLE}")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        archive.extractall(ROOT)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "training/requirements-qlora-v1.txt"],
        check=True,
    )
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from hybrid_agent.evaluation import evaluate_responses
    from hybrid_agent.experiments import verify_experiment

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if compute_capability[0] >= 8 else torch.float16

    verified = verify_experiment(
        ROOT / CONFIG_PATH, workspace=ROOT
    )
    config = verified.config
    train_records = read_jsonl(verified.manifest_path.parent / "train.jsonl")
    validation_records = read_jsonl(verified.manifest_path.parent / "validation.jsonl")
    evaluation_cases = read_jsonl(verified.evaluation_path)
    accepted_records = len(train_records) + len(validation_records)
    run_class = "candidate" if accepted_records >= MINIMUM_MEANINGFUL_RECORDS else "pipeline-only"
    run_dir = ROOT / config["output_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "phase": "verified",
                "gpu": gpu_name,
                "compute_dtype": str(compute_dtype),
                "train_records": len(train_records),
                "validation_records": len(validation_records),
                "run_class": run_class,
            }
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = True

    def generate_response(active_model, prompt: str) -> str:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(active_model.device)
        with torch.inference_mode():
            output = active_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    base_responses = [
        {"id": case["id"], "prompt": case["prompt"], "response": generate_response(model, case["prompt"])}
        for case in evaluation_cases
    ]
    write_jsonl(run_dir / "base-responses.jsonl", base_responses)
    base_result = evaluate_responses(evaluation_cases, base_responses)
    print(json.dumps({"phase": "base-evaluation", "result": base_result}), flush=True)

    def render_record(record: dict) -> str:
        return tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

    train_dataset = Dataset.from_dict({"text": [render_record(row) for row in train_records]})
    validation_dataset = Dataset.from_dict(
        {"text": [render_record(row) for row in validation_records]}
    )
    lora = config["lora"]
    training = config["training"]
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training["gradient_checkpointing"]
    )
    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(run_dir),
        seed=config["seed"],
        max_length=training["max_sequence_length"],
        num_train_epochs=training["epochs"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler"],
        warmup_ratio=training["warmup_ratio"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        gradient_checkpointing=training["gradient_checkpointing"],
        optim=training["optimizer"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        logging_steps=training["logging_steps"],
        eval_strategy="steps",
        eval_steps=training["evaluation_steps"],
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        report_to="none",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()
    trainer.save_model(run_dir / "adapter")
    tokenizer.save_pretrained(run_dir / "adapter")
    print(json.dumps({"phase": "trained", "metrics": train_result.metrics}), flush=True)

    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    adapter_responses = [
        {"id": case["id"], "prompt": case["prompt"], "response": generate_response(model, case["prompt"])}
        for case in evaluation_cases
    ]
    write_jsonl(run_dir / "adapter-responses.jsonl", adapter_responses)
    adapter_result = evaluate_responses(evaluation_cases, adapter_responses)
    adapter_dir = run_dir / "adapter"
    adapter_hashes = {
        str(path.relative_to(run_dir)): file_hash(path)
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }
    registration_eligible = (
        run_class == "candidate"
        and adapter_result["registration_eligible"]
        and adapter_result["score"] > base_result["score"]
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "trained",
        "run_class": run_class,
        "accepted_records": accepted_records,
        "minimum_meaningful_records": MINIMUM_MEANINGFUL_RECORDS,
        "registration_eligible": registration_eligible,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "dataset_manifest_sha256": verified.manifest_sha256,
        "evaluation_suite_sha256": verified.evaluation_sha256,
        "gpu": gpu_name,
        "compute_dtype": str(compute_dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "human_evaluation_required": True,
        "base_score": base_result,
        "adapter_score": adapter_result,
        "training_metrics": train_result.metrics,
        "adapter_sha256": adapter_hashes,
    }
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.make_archive(str(RESULT_ARCHIVE.with_suffix("")), "zip", root_dir=run_dir)
    print(json.dumps({"phase": "complete", "archive": str(RESULT_ARCHIVE)}), flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
