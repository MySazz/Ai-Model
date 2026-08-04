"""Evaluate the pinned Qwen3-8B instruct model before spending GPU time tuning it."""

from __future__ import annotations

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
RUN_DIR = ROOT / "training/runs/qwen3-8b-baseline-v1"
RESULT = Path("/content/qwen3-8b-baseline-v1-results.zip")
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EVALUATION_SUITE = "automated-development-v2"
SYSTEM_PROMPT = ""
RETRIEVAL_DATASET_NAME = ""
RETRIEVAL_TOP_K = 0
MAX_NEW_TOKENS = 256
EXECUTABLE_EVALUATION = False


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    from hybrid_agent.evaluation import evaluate_responses

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    cases = read_jsonl(ROOT / f"datasets/evaluations/{EVALUATION_SUITE}.jsonl")
    retrieval_rows = (
        read_jsonl(ROOT / f"datasets/processed/{RETRIEVAL_DATASET_NAME}/train.jsonl")
        if RETRIEVAL_DATASET_NAME else []
    )

    def retrieve(prompt: str) -> list[dict]:
        if not retrieval_rows or not RETRIEVAL_TOP_K:
            return []
        stop = {"a", "an", "and", "the", "to", "of", "in", "it", "is", "for", "or", "you"}
        wanted = set(re.findall(r"[a-z0-9_]+", prompt.casefold())) - stop
        ranked = []
        for row in retrieval_rows:
            candidate = row["messages"][-2]["content"]
            tokens = set(re.findall(r"[a-z0-9_]+", candidate.casefold())) - stop
            score = len(wanted & tokens) / len(wanted | tokens) if wanted | tokens else 0.0
            ranked.append((score, row["id"], row))
        return [row for _, _, row in sorted(ranked, reverse=True)[:RETRIEVAL_TOP_K]]

    def generate(prompt: str) -> str:
        messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else [])
        for example in reversed(retrieve(prompt)):
            messages.extend(example["messages"][-2:])
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    responses = [
        {"id": case["id"], "prompt": case["prompt"], "response": generate(case["prompt"])}
        for case in cases
    ]
    (RUN_DIR / "responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses), encoding="utf-8"
    )
    if EXECUTABLE_EVALUATION:
        from hybrid_agent.executable_evaluation import run_python_checks
        evaluation = evaluate_responses(
            cases, responses, executable_judge=run_python_checks
        )
    else:
        evaluation = evaluate_responses(cases, responses)
    summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "evaluation_suite": EVALUATION_SUITE,
        "system_prompt": SYSTEM_PROMPT,
        "retrieval_dataset": RETRIEVAL_DATASET_NAME or None,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "gpu": torch.cuda.get_device_name(0),
        "evaluation": evaluation,
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.make_archive(str(RESULT.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(json.dumps({"phase": "complete", "result": summary, "archive": str(RESULT)}), flush=True)


if __name__ == "__main__":
    main()
