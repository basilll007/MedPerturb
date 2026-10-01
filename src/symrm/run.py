"""Single entrypoint for SymRM experiments.

Usage:
    python -m symrm.run --config configs/symrm/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
import torch

from symrm.compute_tracker import ComputeTracker
from symrm.config import SymRMConfig


def run_gate0_verification(cfg: SymRMConfig) -> dict[str, Any]:
    print("=" * 60)
    print("SymRM Gate 0: Hardware & Environment Verification")
    print("=" * 60)

    # 1. PyTorch & CUDA Check
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch_version}")
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        raise RuntimeError("CUDA is not available in PyTorch!")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"GPU device: {device_name}")
    print(f"Compute capability: {capability} (sm_{capability[0]}{capability[1]})")
    print(f"Device arch list: {arch_list}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    # Check Blackwell (sm_120) support
    if capability[0] >= 12 or "sm_120" in arch_list:
        print("[OK] Blackwell (sm_120) compute capability natively supported.")
    else:
        print(f"[WARN] Compute capability is sm_{capability[0]}{capability[1]}, arch list: {arch_list}")

    # 2. BitsAndBytes 4-bit Forward Pass
    import bitsandbytes as bnb
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    print(f"bitsandbytes version: {bnb.__version__}")
    print(f"Testing 4-bit forward pass on: {cfg.model.model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.model.compute_dtype),
        bnb_4bit_quant_type=cfg.model.quant_type,
    )

    tracker = ComputeTracker(
        log_path=cfg.logging.compute_log_path,
        run_id="gate0_env_verify",
        model=cfg.model.model_id,
        condition="forward_pass_4bit",
        seed=cfg.data.seed,
    )
    tracker.start()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_id,
        num_labels=1,
        quantization_config=bnb_config,
        device_map=cfg.model.device_map,
    )

    test_prompt = "Patient: 68-year-old male with eGFR of 24 mL/min/1.73m2 presenting with acute pain."
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    tracker.record_tokens(int(inputs.input_ids.shape[1]))

    with torch.no_grad():
        out = model(**inputs)

    record = tracker.stop()

    print("[OK] 4-bit forward pass completed successfully.")
    print(f"Logits shape: {out.logits.shape}")
    print(f"Logits value: {out.logits.item():.4f}")
    print(f"Peak VRAM: {record['peak_vram_gb']} GB")
    print(f"Tokens/sec: {record['tokens_per_sec']}")
    print(f"Recorded to: {cfg.logging.compute_log_path}")

    # Explicit memory cleanup
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "capability": capability,
        "arch_list": arch_list,
        "total_vram_gb": round(total_vram_gb, 2),
        "bnb_version": bnb.__version__,
        "model_id": cfg.model.model_id,
        "peak_vram_gb": record["peak_vram_gb"],
        "tokens_per_sec": record["tokens_per_sec"],
        "compute_record": record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SymRM Pilot Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--phase", type=str, default=None, help="Override phase from config")
    args = parser.parse_args()

    cfg = SymRMConfig.from_yaml(args.config)
    if args.phase:
        cfg.phase = args.phase

    if cfg.phase == "gate0":
        run_gate0_verification(cfg)
    elif cfg.phase == "phase1_data":
        print(f"Phase 1 Data Generation (Config: {args.config}) - awaiting Gate 1 approval.")
    elif cfg.phase == "phase1_probe":
        print(f"Phase 1 RM Probe (Config: {args.config}) - awaiting Gate 2 approval.")
    elif cfg.phase == "phase2_train":
        print(f"Phase 2 RM Training (Config: {args.config}) - awaiting Gate 3 approval.")
    elif cfg.phase == "phase3_bon":
        print(f"Phase 3 Best-of-N Evaluation (Config: {args.config}) - awaiting Gate 4 approval.")
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}")


if __name__ == "__main__":
    main()
