"""Generate, repair, and execution-filter a training-only candidate corpus on Colab."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


BUNDLE = Path("/content/hybrid-agent-colab-input.zip")
ROOT = Path("/content/hybrid-agent")
RUN_DIR = ROOT / "training/runs/qwen3-4b-execution-feedback-repair-v1"
RESULT = Path("/content/qwen3-4b-execution-feedback-repair-v1-results.zip")
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TASKS = "datasets/generation/execution-guided-tasks-v1.jsonl"
SAMPLES_PER_TASK = 4
MAX_REPAIR_ATTEMPTS = 2
SEED = 20260804
SYSTEM_PROMPT = (
    "You are generating candidate training answers for strict automated validation. Follow the "
    "requested interface exactly. Return only the requested answer. For code, use one complete "
    "Python block with imports and handle failure paths, cleanup, and containment correctly."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from hybrid_agent.candidate_filtering import filter_candidates
    from hybrid_agent.evaluation import evaluate_responses
    from hybrid_agent.executable_evaluation import run_python_checks
    from hybrid_agent.repair_feedback import named_failures, repair_prompt

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.eval()

    def generate(messages: list[dict], seed: int, executable: bool) -> str:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=512 if executable else 192, repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    tasks_path = ROOT / TASKS
    tasks = read_jsonl(tasks_path)
    candidates: list[dict] = []
    repair_trace: list[dict] = []
    for task_index, task in enumerate(tasks):
        executable = task["evaluator"]["type"] == "executable_assertions"
        evaluation_task = dict(task)
        if executable:
            evaluation_task["critical_failure"] = True
        for sample_index in range(SAMPLES_PER_TASK):
            base_seed = SEED + task_index * 100 + sample_index
            base_id = f"{task['id']}-sample-{sample_index}"
            response = generate([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ], base_seed, executable)
            candidate = {
                "id": base_id, "task_id": task["id"], "response": response,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "generation_seed": base_seed,
            }
            candidates.append(candidate)
            for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
                result = evaluate_responses(
                    [evaluation_task], [{"id": task["id"], "response": response}],
                    executable_judge=run_python_checks if executable else None,
                )["cases"][0]
                if result["passed"]:
                    break
                failures = named_failures(result)
                repair_seed = base_seed + attempt * 10_000
                response = generate([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt(
                        task["prompt"], response, failures, attempt
                    )},
                ], repair_seed, executable)
                repair_id = f"{base_id}-repair-{attempt}"
                candidate = {
                    "id": repair_id, "task_id": task["id"], "response": response,
                    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "generation_seed": repair_seed, "parent_candidate_id": base_id,
                    "repair_attempt": attempt, "feedback_checks": failures,
                }
                candidates.append(candidate)
                repair_trace.append({
                    "candidate_id": repair_id, "parent_candidate_id": base_id,
                    "attempt": attempt, "feedback_checks": failures,
                })
            print(json.dumps({
                "phase": "candidate_complete", "task_id": task["id"],
                "sample": sample_index, "candidates": len(candidates),
            }), flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RUN_DIR / "raw-and-repaired-candidates.jsonl"
    accepted_path = RUN_DIR / "accepted-candidates.jsonl"
    trace_path = RUN_DIR / "repair-trace.jsonl"
    write_jsonl(raw_path, candidates)
    write_jsonl(trace_path, repair_trace)
    accepted, report = filter_candidates(
        tasks, candidates, executable_judge=run_python_checks,
        minimum_per_task=2, near_duplicate_threshold=0.82, maximum_capability_ratio=2.0,
        output_license=os.environ.get("HYBRID_AGENT_OUTPUT_LICENSE"),
        license_basis=os.environ.get("HYBRID_AGENT_LICENSE_BASIS"),
    )
    write_jsonl(accepted_path, accepted)
    report.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "samples_per_task": SAMPLES_PER_TASK, "maximum_repair_attempts": MAX_REPAIR_ATTEMPTS,
        "generation_seed": SEED, "tasks_sha256": digest(tasks_path),
        "raw_candidates_sha256": digest(raw_path), "repair_trace_sha256": digest(trace_path),
        "accepted_candidates_sha256": digest(accepted_path),
        "repair_candidates": len(repair_trace), "gpu": torch.cuda.get_device_name(0),
    })
    (RUN_DIR / "filter-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({
        "phase": "complete", "accepted": len(accepted), "input": len(candidates),
        "repairs": len(repair_trace), "ready_for_training": report["ready_for_training"],
        "archive": str(RESULT),
    }), flush=True)


if __name__ == "__main__":
    main()
