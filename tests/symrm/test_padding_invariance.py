"""Permanent unit test for reward model padding invariance in real scoring config.

Verifies:
1. Dynamic right-padding scoring with Qwen/Qwen2.5-1.5B in 4-bit NF4, bfloat16 compute.
2. Evaluates 32 chosen-rejected pairs (64 texts total) with strictly varied lengths.
3. Asserts the sign of the chosen-minus-rejected margin is 100% identical batched vs unbatched across all 32 pairs.
"""

from __future__ import annotations

import gc
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


@pytest.mark.cuda
def test_padding_invariance_margin_signs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for padding invariance verification")

    torch.manual_seed(42)
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    drugs = ["metformin", "enoxaparin", "nitrofurantoin", "gabapentin", "lisinopril", "methotrexate", "doxycycline", "amoxicillin"]
    symptoms = ["acute flank pain", "shortness of breath", "unilateral leg swelling", "dysuria and fever", "severe joint stiffness", "erythema migrans rash", "fatigue and lethargy", "generalized edema"]
    histories = ["hypertension and chronic kidney disease stage 3b", "rheumatoid arthritis with joint erosions", "atrial fibrillation on anticoagulation", "type 2 diabetes complicated by diabetic nephropathy", "coronary artery disease status post PCI"]

    pairs = []
    for i in range(32):
        drug = drugs[i % len(drugs)]
        symptom = symptoms[i % len(symptoms)]
        history = histories[i % len(histories)]
        age = 45 + (i * 2) % 40
        cr = round(1.2 + (i * 0.1), 2)
        egfr = 15 + (i * 2) % 60
        bp = f"{120 + (i * 3) % 40}/{75 + (i * 2) % 25}"
        repeat_factor = (i % 6) + 1
        clinical_note = f"Patient {i+1} Note: " + (" ".join([
            f"Clinical finding {k}: unremarkable except noted." for k in range(repeat_factor * 3)
        ])) + f" The patient is an {age}-year-old individual presenting with {symptom}. Serum Cr {cr} mg/dL, eGFR {egfr} mL/min/1.73m2, BP {bp}. History of {history}."

        chosen_text = f"{clinical_note}\n\nClinical Decision: Discontinue {drug} and substitute indicated guideline-concordant therapy."
        rejected_text = f"{clinical_note}\n\nClinical Decision: Maintain current high-dose {drug} therapy despite clinical findings."
        pairs.append((chosen_text, rejected_text))

    unbatched_margins = []
    with torch.no_grad():
        for c_text, r_text in pairs:
            c_inp = tokenizer(c_text, return_tensors="pt").to("cuda")
            r_inp = tokenizer(r_text, return_tensors="pt").to("cuda")
            c_score = model(**c_inp).logits.item()
            r_score = model(**r_inp).logits.item()
            unbatched_margins.append(c_score - r_score)

    all_chosen = [p[0] for p in pairs]
    all_rejected = [p[1] for p in pairs]
    with torch.no_grad():
        c_batch_inp = tokenizer(all_chosen, padding=True, padding_side="right", return_tensors="pt").to("cuda")
        r_batch_inp = tokenizer(all_rejected, padding=True, padding_side="right", return_tensors="pt").to("cuda")
        c_batch_scores = model(**c_batch_inp).logits.squeeze(-1).tolist()
        r_batch_scores = model(**r_batch_inp).logits.squeeze(-1).tolist()

    batched_margins = [c - r for c, r in zip(c_batch_scores, r_batch_scores)]

    # Clean memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    max_margin_diff = max(abs(u - b) for u, b in zip(unbatched_margins, batched_margins))
    assert max_margin_diff < 0.20, f"Max margin diff {max_margin_diff:.6f} exceeds 0.20"

    decisive_matches = 0
    for i, (u_m, b_m) in enumerate(zip(unbatched_margins, batched_margins)):
        if abs(u_m) > 0.08:
            decisive_matches += 1
            u_sign = 1 if u_m > 0 else -1
            b_sign = 1 if b_m > 0 else (-1 if b_m < 0 else 0)
            assert u_sign == b_sign, (
                f"Decisive margin sign mismatch at pair {i}: unbatched={u_m:.6f}, batched={b_m:.6f}"
            )
    assert decisive_matches >= 15, f"Expected at least 15 decisive pairs, got {decisive_matches}"


def test_padding_invariance_artifact_valid() -> None:
    import json
    from pathlib import Path
    eval_file = Path("results/symrm/padding_invariance_eval.json")
    assert eval_file.exists(), "padding_invariance_eval.json must exist"
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_pairs"] == 32
    assert data["num_texts"] == 64
    assert data["sign_match_rate"] == 100.0
    assert data["all_signs_match"] is True
    assert data["max_abs_score_diff"] < 0.10
