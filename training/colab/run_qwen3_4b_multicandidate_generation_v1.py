"""Generate and execution-filter a training-only candidate corpus on Colab."""

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
RUN_DIR = ROOT / "training/runs/qwen3-4b-multicandidate-generation-v1"
RESULT = Path("/content/qwen3-4b-multicandidate-generation-v1-results.zip")
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TASKS = "datasets/generation/execution-guided-tasks-v1.jsonl"
SAMPLES_PER_TASK = 4
SEED = 20260804
SYSTEM_PROMPT = (
    "You are generating candidate training answers for strict automated validation. Follow the "
    "requested interface exactly. For code, return one complete Python block with imports and "
    "handle failure paths, cleanup, boolean results, and path containment correctly. For policy "
    "answers, be concise, refuse unsafe or unverifiable requests, and offer a safe next step."
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
    from hybrid_agent.executable_evaluation import run_python_checks

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
    tasks_path = ROOT / TASKS
    tasks = read_jsonl(tasks_path)
    candidates = []
    for task_index, task in enumerate(tasks):
        executable = task["evaluator"]["type"] == "executable_assertions"
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        for sample_index in range(SAMPLES_PER_TASK):
            generation_seed = SEED + task_index * 100 + sample_index
            torch.manual_seed(generation_seed)
            torch.cuda.manual_seed_all(generation_seed)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=512 if executable else 192,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            candidates.append({
                "id": f"{task['id']}-sample-{sample_index}",
                "task_id": task["id"],
                "response": response,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "generation_seed": generation_seed,
            })
            print(json.dumps({
                "phase": "generated", "completed": len(candidates),
                "total": len(tasks) * SAMPLES_PER_TASK, "task_id": task["id"],
            }), flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RUN_DIR / "raw-candidates.jsonl"
    accepted_path = RUN_DIR / "accepted-candidates.jsonl"
    write_jsonl(raw_path, candidates)
    accepted, report = filter_candidates(
        tasks, candidates, executable_judge=run_python_checks,
        minimum_per_task=2, near_duplicate_threshold=0.82,
        maximum_capability_ratio=2.0,
        output_license=os.environ.get("HYBRID_AGENT_OUTPUT_LICENSE"),
        license_basis=os.environ.get("HYBRID_AGENT_LICENSE_BASIS"),
    )
    write_jsonl(accepted_path, accepted)
    report.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "samples_per_task": SAMPLES_PER_TASK,
        "generation_seed": SEED,
        "temperature": 0.7,
        "top_p": 0.9,
        "tasks_sha256": digest(tasks_path),
        "raw_candidates_sha256": digest(raw_path),
        "accepted_candidates_sha256": digest(accepted_path),
        "gpu": torch.cuda.get_device_name(0),
    })
    (RUN_DIR / "filter-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({
        "phase": "complete", "accepted": len(accepted),
        "input": len(candidates), "ready_for_training": report["ready_for_training"],
        "archive": str(RESULT),
    }), flush=True)


if __name__ == "__main__":
    main()
