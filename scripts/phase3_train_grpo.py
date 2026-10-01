"""Phase 3 Post-Training Pipeline: GRPO with QLoRA using Unsloth.

Supports two comparable policies under identical base configuration:
1. Policy A: Correctness-only reward (R_correct)
2. Policy B: Neuro-symbolic reward (R_correct + lambda * R_symbolic)

Conservative 8 GB VRAM design:
- 4-bit quantized base model
- LoRA rank 16 on all linear layers
- Sequence length <= 512 tokens (prompts <= 384, completions <= 64)
- Group size G = 4 rollouts (G = 2 for smoke test)
- Gradient accumulation = 4, paged_adamw_8bit optimizer
- Unsloth gradient checkpointing
"""

from __future__ import annotations

import unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainerCallback

from medperturb.perturbations.registry import CONDITIONS, build_examples
from medperturb.prompts.templates import build_prompt
from medperturb.reward.batch import (
    create_reward_symbolic_func,
    reward_correct_func,
    reward_format_func,
)
from medperturb.schemas.response import Example

MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
TRAIN_PARQUET = ROOT / "data" / "processed" / "phase3" / "train.parquet"
OUTPUT_ROOT = ROOT / "results" / "phase3"


class MetricsTrackerCallback(TrainerCallback):
    """Logs training metrics, rewards, losses, and VRAM to CSV at every step."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.history: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        vram_gb = float(torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        vram_peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vram_allocated_gb": round(vram_gb, 3),
            "vram_peak_gb": round(vram_peak_gb, 3),
            **logs,
        }
        self.history.append(entry)
        pd.DataFrame(self.history).to_csv(self.log_path, index=False)


def prepare_training_dataset(parquet_path: Path, tokenizer, max_questions: int | None = None) -> Dataset:
    """Prepares and formats training prompts and ground truth examples for GRPO."""
    df = pd.read_parquet(parquet_path)
    by_q = {qid: g for qid, g in df.groupby("question_id")}
    qids = list(by_q.keys())
    if max_questions is not None:
        qids = qids[:max_questions]

    records = []
    # Training conditions: baseline + invariance + adaptation
    train_conditions = ["mcq", "roman_numeral", "fixed_pos", "no_symbols", "none_of_the_provided", "incorrect"]

    for qid in qids:
        sub = by_q[qid]
        examples = build_examples(sub)
        for cond in train_conditions:
            if cond not in examples:
                continue
            ex = examples[cond]
            prompt_text = build_prompt(ex)

            messages = [{"role": "user", "content": prompt_text}]
            # Apply Qwen chat template without thinking tokens
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # Store serialized example data for reward evaluators
            ex_data = {
                "question_id": ex.question_id,
                "source_dataset": ex.source_dataset,
                "source_id": ex.source_id,
                "condition": ex.condition,
                "question": ex.question,
                "option_labels": ex.option_labels,
                "option_texts": ex.option_texts,
                "options_displayed": ex.options_displayed,
                "gold_label": ex.gold_label,
                "gold_text": ex.gold_text,
                "gold_index": ex.gold_index,
                "gold_indices": ex.gold_indices,
            }

            records.append({
                "prompt": formatted_prompt,
                "example_json": json.dumps(ex_data),
                "question_id": ex.question_id,
                "condition": ex.condition,
            })

    dataset = Dataset.from_list(records)
    dataset = dataset.shuffle(seed=42)
    return dataset


def train_policy(
    policy: str,
    lambda_symbolic: float = 1.0,
    beta_format: float = 0.0,
    smoke_test: bool = False,
    max_steps: int = 60,
    seed: int = 42,
) -> Path:
    from trl import GRPOConfig, GRPOTrainer

    if smoke_test:
        policy_name = f"smoke_test_{policy}"
        actual_max_steps = 3
        num_generations = 2
        grad_accum = 2
        max_questions = 4
    else:
        policy_name = f"policy_{policy}"
        actual_max_steps = max_steps
        num_generations = 4
        grad_accum = 4
        max_questions = None

    ckpt_dir = OUTPUT_ROOT / "checkpoints" / policy_name
    training_log_dir = OUTPUT_ROOT / "training" / policy_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    training_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting GRPO Post-Training ===")
    print(f"Policy: {policy} (Run: {policy_name})")
    print(f"Lambda Symbolic: {lambda_symbolic}, Beta Format: {beta_format}")
    print(f"Max Steps: {actual_max_steps}, Num Generations: {num_generations}")
    print(f"Output Directory: {ckpt_dir}")

    # Load 4-bit model with LoRA via Unsloth
    print("[*] Loading Qwen3-4B 4-bit model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=512,
        load_in_4bit=True,
        fast_inference=False,
    )

    print("[*] Applying LoRA adapter (r=16, alpha=16)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # Prepare dataset
    print("[*] Building training dataset from train.parquet...")
    train_dataset = prepare_training_dataset(TRAIN_PARQUET, tokenizer, max_questions=max_questions)
    print(f"    Total training examples: {len(train_dataset)}")

    # Configure reward functions
    if policy == "correctness":
        reward_funcs = [reward_correct_func]
    elif policy == "neurosymbolic":
        reward_funcs = [
            reward_correct_func,
            create_reward_symbolic_func(lambda_symbolic=lambda_symbolic),
        ]
        if beta_format > 0.0:
            reward_funcs.append(reward_format_func)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # Configure GRPO training arguments
    training_args = GRPOConfig(
        output_dir=str(ckpt_dir),
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=384,
        max_completion_length=64,
        max_steps=actual_max_steps,
        save_steps=20,
        save_total_limit=3,
        seed=seed,
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    metrics_cb = MetricsTrackerCallback(training_log_dir / "training_history.csv")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[metrics_cb],
    )

    # Check for existing checkpoints to support restartable training
    last_ckpt = None
    ckpts = sorted(list(ckpt_dir.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
    if ckpts and not smoke_test:
        last_ckpt = str(ckpts[-1])
        print(f"[*] Found checkpoint, resuming training from: {last_ckpt}")

    t0 = time.time()
    print("[*] Commencing GRPO training loop...")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    total_training_time = time.time() - t0

    print(f"[+] Training completed in {total_training_time:.1f}s")

    # Save final adapter weights and tokenizer
    print(f"[*] Saving adapter weights to {ckpt_dir}...")
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Save config and summary manifest
    manifest = {
        "policy": policy,
        "policy_name": policy_name,
        "smoke_test": smoke_test,
        "base_model": MODEL_ID,
        "lambda_symbolic": lambda_symbolic,
        "beta_format": beta_format,
        "max_steps": actual_max_steps,
        "num_generations": num_generations,
        "gradient_accumulation_steps": grad_accum,
        "seed": seed,
        "total_training_time_s": round(total_training_time, 2),
        "peak_vram_gb": round(float(torch.cuda.max_memory_allocated() / (1024**3)), 3) if torch.cuda.is_available() else 0.0,
        "metrics": train_result.metrics,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(training_log_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Training artifacts saved to {ckpt_dir} and {training_log_dir}")
    return ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="MedPerturb Phase 3 GRPO Training.")
    parser.add_argument("--policy", type=str, required=True, choices=["correctness", "neurosymbolic"])
    parser.add_argument("--lambda_symbolic", type=float, default=1.0)
    parser.add_argument("--beta_format", type=float, default=0.0)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_policy(
        policy=args.policy,
        lambda_symbolic=args.lambda_symbolic,
        beta_format=args.beta_format,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
