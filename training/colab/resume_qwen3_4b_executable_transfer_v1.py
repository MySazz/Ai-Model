"""Resume final gates from the saved executable-transfer-v1 adapter."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path("/content/hybrid-agent")
RUN_DIR = ROOT / "training/runs/qwen3-4b-executable-transfer-v1"
RESULT = Path("/content/qwen3-4b-executable-transfer-v1-results.zip")
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SYSTEM_PROMPT = (
    "You are a precise, evidence-grounded Python software agent. Follow requested signatures "
    "exactly. Return complete code with imports. Preserve prior state on failure, validate before "
    "side effects, enforce canonical path containment, clean temporary artifacts, record outcomes "
    "honestly, and never claim unobserved execution."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def mean_recall(rows: list[dict]) -> float:
    return sum(row["target_token_recall"] for row in rows) / len(rows)


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from hybrid_agent.evaluation import evaluate_responses
    from hybrid_agent.executable_evaluation import run_python_checks

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model = PeftModel.from_pretrained(base, RUN_DIR / "adapter")
    model.eval()

    def generate(prompt: str) -> str:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    executable_cases = read_jsonl(ROOT / "datasets/evaluations/executable-development-v1.jsonl")
    executable_responses = read_jsonl(RUN_DIR / "transfer-responses.jsonl")
    executable = evaluate_responses(
        executable_cases, executable_responses, executable_judge=run_python_checks
    )
    policy_cases = read_jsonl(ROOT / "datasets/evaluations/automated-development-v2.jsonl")
    policy_responses = [
        {"id": case["id"], "prompt": case["prompt"], "response": generate(case["prompt"])}
        for case in policy_cases
    ]
    write_jsonl(RUN_DIR / "secondary-transfer-responses.jsonl", policy_responses)
    policy = evaluate_responses(policy_cases, policy_responses)

    base_probes = read_jsonl(RUN_DIR / "base-probes.jsonl")
    adapter_probes = read_jsonl(RUN_DIR / "adapter-probes.jsonl")
    base_heldout = read_jsonl(RUN_DIR / "base-heldout-probes.jsonl")
    adapter_heldout = read_jsonl(RUN_DIR / "adapter-heldout-probes.jsonl")
    base_recall, adapter_recall = mean_recall(base_probes), mean_recall(adapter_probes)
    base_heldout_recall = mean_recall(base_heldout)
    adapter_heldout_recall = mean_recall(adapter_heldout)
    executable_passed = executable["score"] >= 0.75 and not executable["critical_failures"]
    policy_passed = policy["score"] >= 10 / 12 and not policy["critical_failures"]
    heldout_passed = (
        adapter_heldout_recall >= 0.30
        and adapter_heldout_recall >= base_heldout_recall + 0.03
    )
    passed = (
        adapter_recall >= 0.40 and adapter_recall >= base_recall + 0.05
        and heldout_passed and executable_passed and policy_passed
    )
    trainer_state_paths = sorted((RUN_DIR / "checkpoints").glob("checkpoint-*/trainer_state.json"))
    trainer_state = json.loads(trainer_state_paths[-1].read_text()) if trainer_state_paths else {}
    summary = {
        "schema_version": 1, "status": "passed" if passed else "failed",
        "resumed_after_cli_timeout": True, "assistant_mask_verified": True,
        "train_records": 24, "validation_records": 4,
        "base_probe_recall": base_recall, "adapter_probe_recall": adapter_recall,
        "base_heldout_recall": base_heldout_recall,
        "adapter_heldout_recall": adapter_heldout_recall,
        "heldout_passed": heldout_passed,
        "transfer_evaluation": executable, "transfer_passed": executable_passed,
        "secondary_evaluation_suite": "automated-development-v2",
        "secondary_transfer_evaluation": policy,
        "secondary_transfer_passed": policy_passed,
        "best_validation_loss": trainer_state.get("best_metric"),
        "gpu": torch.cuda.get_device_name(0),
    }
    (RUN_DIR / "diagnostic-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({"phase": "complete", "result": summary, "archive": str(RESULT)}), flush=True)


if __name__ == "__main__":
    main()
