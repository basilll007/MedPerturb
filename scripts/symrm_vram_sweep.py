"""Training-step VRAM sweep and thermal profiling for SymRM.

Tests Qwen2.5-1.5B with 4-bit NF4 quantization, LoRA (r=16, all attention + MLP projections),
gradient checkpointing ON, and PagedAdamW 8-bit optimizer across per-device batch sizes 1, 2, 4, 8.
Evaluates 5 full optimizer steps with gradient accumulation to reach an effective batch of 32 pairs
at sequence length 512, tracking peak VRAM, tokens/sec, step time, GPU temperature, and clock speed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SEQ_LEN = 512
EFFECTIVE_BATCH_SIZE = 32
NUM_STEPS = 5
BATCH_SIZES = [1, 2, 4, 8]
SEED = 42

COMPUTE_LOG_PATH = ROOT / "results" / "symrm" / "compute_log.csv"
SWEEP_JSON_PATH = ROOT / "results" / "symrm" / "vram_sweep_results.json"


def query_gpu_telemetry() -> dict[str, float]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = res.stdout.strip().split("\n")[0]
        temp_str, clock_str = [x.strip() for x in line.split(",")]
        return {
            "temperature_c": float(temp_str),
            "clock_mhz": float(clock_str),
        }
    except Exception as e:
        return {"temperature_c": -1.0, "clock_mhz": -1.0}


def log_to_compute_csv(
    run_id: str,
    model: str,
    condition: str,
    seed: int,
    peak_vram_gb: float,
    tokens_per_sec: float,
    wall_clock_min: float,
    gpu_hours: float,
) -> None:
    COMPUTE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "model",
        "condition",
        "seed",
        "peak_vram_gb",
        "tokens_per_sec",
        "wall_clock_min",
        "gpu_hours",
    ]
    file_exists = COMPUTE_LOG_PATH.exists() and COMPUTE_LOG_PATH.stat().st_size > 0
    with open(COMPUTE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "model": model,
            "condition": condition,
            "seed": seed,
            "peak_vram_gb": round(peak_vram_gb, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "wall_clock_min": round(wall_clock_min, 3),
            "gpu_hours": round(gpu_hours, 5),
        })


def run_sweep() -> list[dict[str, Any]]:
    print("=" * 65)
    print("SymRM Task A: Training-Step VRAM Sweep & Thermal Profiling")
    print(f"Model: {MODEL_ID} (4-bit NF4, LoRA r=16, Grad Checkpointing ON)")
    print(f"Seq Len: {SEQ_LEN}, Effective Batch Size: {EFFECTIVE_BATCH_SIZE} pairs, Steps: {NUM_STEPS}")
    print("=" * 65)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="SEQ_CLS",
    )

    results = []

    for bs in BATCH_SIZES:
        grad_accum_steps = EFFECTIVE_BATCH_SIZE // bs
        print(f"\n--- Testing Batch Size: {bs} (Grad Accum: {grad_accum_steps} -> Eff Batch: {EFFECTIVE_BATCH_SIZE}) ---")

        # Clean slate before loading
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_telem = query_gpu_telemetry()
        print(f"Initial GPU Temp: {initial_telem['temperature_c']} C, Clock: {initial_telem['clock_mhz']} MHz")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=1,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        # Enable gradient checkpointing and prepare for kbit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)
        model.train()

        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=2e-4)

        # Generate deterministic synthetic batches
        gen_rng = torch.Generator(device="cuda").manual_seed(SEED + bs)
        vocab_size = len(tokenizer)

        step_times = []
        t0_total = time.perf_counter()

        oom_occurred = False
        try:
            for step in range(NUM_STEPS):
                t0_step = time.perf_counter()
                optimizer.zero_grad()

                for micro in range(grad_accum_steps):
                    # Random token IDs representing chosen and rejected sequences of length SEQ_LEN
                    chosen_ids = torch.randint(0, vocab_size, (bs, SEQ_LEN), device="cuda", generator=gen_rng)
                    chosen_mask = torch.ones((bs, SEQ_LEN), dtype=torch.int32, device="cuda")
                    rejected_ids = torch.randint(0, vocab_size, (bs, SEQ_LEN), device="cuda", generator=gen_rng)
                    rejected_mask = torch.ones((bs, SEQ_LEN), dtype=torch.int32, device="cuda")

                    r_chosen = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits.squeeze(-1)
                    r_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)

                    # Bradley-Terry loss: -log(sigmoid(r_c - r_r))
                    loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
                    scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()

                optimizer.step()
                torch.cuda.synchronize()
                step_duration = time.perf_counter() - t0_step
                step_times.append(step_duration)
                print(f"  Step {step + 1}/{NUM_STEPS} - Time: {step_duration:.3f}s - Loss: {loss.item():.4f}")

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Batch size {bs} exceeded available VRAM!")
            oom_occurred = True

        total_time_sec = time.perf_counter() - t0_total
        final_telem = query_gpu_telemetry()
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        if not oom_occurred:
            avg_step_time = sum(step_times) / len(step_times)
            # Each step processes EFFECTIVE_BATCH_SIZE pairs * 2 sequences * SEQ_LEN tokens
            tokens_per_step = EFFECTIVE_BATCH_SIZE * 2 * SEQ_LEN
            tokens_per_sec = tokens_per_step / avg_step_time
            wall_clock_min = total_time_sec / 60.0
            gpu_hours = total_time_sec / 3600.0

            under_7gb = peak_vram_gb < 7.0
            print(f"[OK] BS {bs}: Peak VRAM = {peak_vram_gb:.3f} GB (<7.0GB: {under_7gb})")
            print(f"     Avg Step Time = {avg_step_time:.3f}s, Throughput = {tokens_per_sec:.1f} tokens/s")
            print(f"     Final Temp = {final_telem['temperature_c']} C, Clock = {final_telem['clock_mhz']} MHz")

            log_to_compute_csv(
                run_id=f"vram_sweep_bs{bs}",
                model=MODEL_ID,
                condition=f"bt_train_bs{bs}_ga{grad_accum_steps}",
                seed=SEED,
                peak_vram_gb=peak_vram_gb,
                tokens_per_sec=tokens_per_sec,
                wall_clock_min=wall_clock_min,
                gpu_hours=gpu_hours,
            )

            results.append({
                "batch_size": bs,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "peak_vram_gb": round(peak_vram_gb, 3),
                "under_7gb": under_7gb,
                "avg_step_time_s": round(avg_step_time, 3),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "total_time_s": round(total_time_sec, 2),
                "temp_start_c": initial_telem["temperature_c"],
                "temp_end_c": final_telem["temperature_c"],
                "clock_start_mhz": initial_telem["clock_mhz"],
                "clock_end_mhz": final_telem["clock_mhz"],
                "status": "PASS",
            })
        else:
            results.append({
                "batch_size": bs,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "peak_vram_gb": round(peak_vram_gb, 3),
                "under_7gb": False,
                "status": "OOM",
            })

        # Teardown model
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(SWEEP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Full sweep results saved to {SWEEP_JSON_PATH}")
    return results


if __name__ == "__main__":
    run_sweep()
