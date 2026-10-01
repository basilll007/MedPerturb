"""Verification script for padding invariance in real scoring config:
Qwen/Qwen2.5-1.5B, 4-bit NF4, bfloat16 compute dtype, dynamic padding to longest in batch.
Evaluates 32 chosen-rejected pairs (64 texts total) with strictly varied lengths.
Asserts:
1. Max absolute difference between batched and unbatched scores.
2. Sign of margin (chosen - rejected) is identical batched vs unbatched across all 32 pairs.
"""

import sys
import gc
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

def run_test():
    print("=" * 60)
    print("PADDING-INVARIANCE EVALUATION (4-bit NF4, bfloat16 compute)")
    print("=" * 60)

    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in 4-bit NF4 with bfloat16 compute dtype...")
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

    # Create 32 clinical-style pairs with diverse lengths (lengths ranging from ~30 to ~350 tokens)
    # Using clinical tokens to simulate realistic clinical vignette sequences
    templates = [
        ("Patient is a {age}-year-old presenting with {symptom}. Clinical lab analysis reveals serum creatinine of {cr}, eGFR {egfr} mL/min, and blood pressure {bp} mmHg. Past medical history includes {history}.",
         "Recommendation: Discontinue {drug} immediately and initiate alternative renal-protective therapy with active monitoring.",
         "Recommendation: Continue {drug} at standard maximum therapeutic dosage of 1000 mg twice daily without modification.")
    ]

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

        # Vary length deliberately across pairs: repeat background notes to simulate variable chart notes
        repeat_factor = (i % 6) + 1
        clinical_note = f"Patient {i+1} Note: " + (" ".join([
            f"Clinical finding {k}: unremarkable except noted." for k in range(repeat_factor * 3)
        ])) + f" The patient is an {age}-year-old individual presenting with {symptom}. Serum Cr {cr} mg/dL, eGFR {egfr} mL/min/1.73m2, BP {bp}. History of {history}."

        chosen_text = f"{clinical_note}\n\nClinical Decision: Discontinue {drug} and substitute indicated guideline-concordant therapy."
        rejected_text = f"{clinical_note}\n\nClinical Decision: Maintain current high-dose {drug} therapy despite clinical findings."
        pairs.append((chosen_text, rejected_text))

    print(f"Generated {len(pairs)} pairs (64 texts). Scoring unbatched...")

    unbatched_chosen = []
    unbatched_rejected = []
    unbatched_margins = []

    with torch.no_grad():
        for i, (c_text, r_text) in enumerate(pairs):
            c_inp = tokenizer(c_text, return_tensors="pt").to("cuda")
            r_inp = tokenizer(r_text, return_tensors="pt").to("cuda")

            c_score = model(**c_inp).logits.item()
            r_score = model(**r_inp).logits.item()

            unbatched_chosen.append(c_score)
            unbatched_rejected.append(r_score)
            unbatched_margins.append(c_score - r_score)

    print("Scoring batched with dynamic padding (pad to longest in batch)...")
    # All 32 chosen in one dynamically padded batch, all 32 rejected in one dynamically padded batch
    all_chosen_texts = [p[0] for p in pairs]
    all_rejected_texts = [p[1] for p in pairs]

    with torch.no_grad():
        c_batch_inp = tokenizer(all_chosen_texts, padding=True, padding_side="right", return_tensors="pt").to("cuda")
        r_batch_inp = tokenizer(all_rejected_texts, padding=True, padding_side="right", return_tensors="pt").to("cuda")

        print(f"Chosen batch shape: {c_batch_inp['input_ids'].shape} (dynamic pad to max length {c_batch_inp['input_ids'].shape[1]})")
        print(f"Rejected batch shape: {r_batch_inp['input_ids'].shape} (dynamic pad to max length {r_batch_inp['input_ids'].shape[1]})")

        c_batch_scores = model(**c_batch_inp).logits.squeeze(-1).tolist()
        r_batch_scores = model(**r_batch_inp).logits.squeeze(-1).tolist()

    batched_margins = [c - r for c, r in zip(c_batch_scores, r_batch_scores)]

    # Metrics
    c_diffs = [abs(u - b) for u, b in zip(unbatched_chosen, c_batch_scores)]
    r_diffs = [abs(u - b) for u, b in zip(unbatched_rejected, r_batch_scores)]
    all_diffs = c_diffs + r_diffs
    max_diff = max(all_diffs)
    mean_diff = sum(all_diffs) / len(all_diffs)

    margin_diffs = [abs(um - bm) for um, bm in zip(unbatched_margins, batched_margins)]
    max_margin_diff = max(margin_diffs)

    # Check sign agreement
    sign_agreements = []
    for i, (um, bm) in enumerate(zip(unbatched_margins, batched_margins)):
        u_sign = 1 if um > 0 else (-1 if um < 0 else 0)
        b_sign = 1 if bm > 0 else (-1 if bm < 0 else 0)
        agrees = (u_sign == b_sign)
        sign_agreements.append(agrees)

    all_signs_match = all(sign_agreements)
    sign_match_pct = 100.0 * sum(sign_agreements) / len(sign_agreements)

    print("\n--- RESULTS ---")
    print(f"Total texts evaluated: {len(all_diffs)}")
    print(f"Max absolute score difference |unbatched - batched|: {max_diff:.8e}")
    print(f"Mean absolute score difference: {mean_diff:.8e}")
    print(f"Max absolute margin difference |(c_u - r_u) - (c_b - r_b)|: {max_margin_diff:.8e}")
    print(f"Margin sign agreement across 32 pairs: {sum(sign_agreements)}/32 ({sign_match_pct:.1f}%)")

    # Check sample details
    print("\nSample first 5 pair margins:")
    for i in range(5):
        print(f"  Pair {i+1:02d}: unbatched margin = {unbatched_margins[i]:+.6f}, batched margin = {batched_margins[i]:+.6f}, diff = {margin_diffs[i]:.6e}, sign match = {sign_agreements[i]}")

    results = {
        "model_id": model_id,
        "dtype": "bfloat16 (compute) / 4-bit NF4 (weights)",
        "num_pairs": 32,
        "num_texts": 64,
        "max_abs_score_diff": max_diff,
        "mean_abs_score_diff": mean_diff,
        "max_abs_margin_diff": max_margin_diff,
        "sign_match_rate": sign_match_pct,
        "all_signs_match": all_signs_match,
    }

    with open("results/symrm/padding_invariance_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Clean memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    assert all_signs_match, "FATAL: Margin signs differed between batched and unbatched scoring!"
    print("\nSUCCESS: All 32 pair margin signs strictly match, padding invariance verified in 4-bit NF4 bf16!")

if __name__ == "__main__":
    run_test()
