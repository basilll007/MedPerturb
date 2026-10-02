# SymRM Pilot Pre-Registration Document

**Study Title:** Symbolic Supervision for Neural Reward Models: Evaluating Generalization to Decisive Clinical Evidence  
**Target Venue:** NAACL 2027 Main Track  
**Principal Investigator / Research Engineering Team:** MedPerturb Lead Research Engineer  
**Date:** 2026-10-01  
**Status:** Pre-Registered (Prior to Model Probing & Post-Training)  
**Data Version Tag:** `v0-unreviewed` (Human Clinical Review Pending)

---

## 1. Research Question & Primary Hypotheses

### Research Question
Can symbolic supervision teach neural reward models to generalize beyond the rules they were trained on?

### Core Hypotheses
1. **Hypothesis 1 (Primary - Out-of-Distribution Generalization):**  
   Neural reward models trained with symbolic supervision (C4: Bradley-Terry loss + Pair Consistency loss + Symbolic Auxiliary Heads) achieve significantly higher **Discriminative Flip Rate (DFR)** on completely held-out clinical rule families (**Far-OOD: Drug Allergy**) than standard outcome-supervised reward models (C1: Outcome Bradley-Terry loss only).
2. **Diagnostic Metric 1 (Resistance to Keyword Shortcuts):**  
   Standard outcome-trained reward models (C1) exhibit elevated **Over-flip Rates** on near-miss clinical items. Pair consistency and auxiliary symbolic supervision suppress this.
3. **Diagnostic Metric 2 (Paraphrase Noise Floor):**  
   Based on findings in Yang et al. (arXiv:2605.01048), surface-form modifications can induce spurious metric changes. We implement a **Null-Edit** item type (paraphrasing non-decisive sentences/fields) to establish a baseline noise floor. We hypothesize that symbolic supervision reduces spurious flips on null-edits compared to baseline models.
4. **Secondary Hypothesis (Counterfactual Invariance):**  
   Symbolic auxiliary heads enforce decisive evidence override internally, producing higher **Counterfactual Flip Rate (CFR)** across both seen (Train), near-transfer (Near-OOD), and novel (Far-OOD) clinical scenarios, when adjusted by the null-edit flip rate.

---

## 2. Experimental Conditions (2 $\times$ 2 Factorial Design)

Base model: `Qwen/Qwen2.5-1.5B`, 4-bit NF4 quantized, `bfloat16` compute dtype, LoRA ($r=16, \alpha=16$, all attention and MLP projections), 8-bit PagedAdamW optimizer.

- **C1 (Baseline - Outcome Bradley-Terry):** Standard pairwise loss on chosen vs. rejected prescriptions:
  $$\mathcal{L}_{\text{C1}} = -\log \sigma(r(x, y_c) - r(x, y_r))$$
- **C2 (Counterfactual Margin Consistency):** Outcome BT loss + pairwise margin consistency loss enforcing that the margin flips sign under the counterfactual intervention:
  $$\mathcal{L}_{\text{C2}} = \mathcal{L}_{\text{C1}} + \lambda_{\text{pair}} \mathcal{L}_{\text{consistency}}$$
- **C3 (Symbolic Auxiliary Supervision):** Outcome BT loss + auxiliary classification heads predicting `rule_fired` (binary) and `decisive_fact_type` (multiclass) from internal representations:
  $$\mathcal{L}_{\text{C3}} = \mathcal{L}_{\text{C1}} + \lambda_{\text{aux}} (\mathcal{L}_{\text{fired}} + \mathcal{L}_{\text{fact}})$$
- **C4 (Proposed - Combined Method):** Full objective combining outcome BT loss, pair consistency loss, and symbolic auxiliary prediction heads:
  $$\mathcal{L}_{\text{C4}} = \mathcal{L}_{\text{C1}} + \lambda_{\text{pair}} \mathcal{L}_{\text{consistency}} + \lambda_{\text{aux}} (\mathcal{L}_{\text{fired}} + \mathcal{L}_{\text{fact}})$$

*Note on inference:* In all conditions (C1–C4), all auxiliary heads and symbolic verifiers are strictly detached during evaluation. The reward model scores candidate responses solely via its scalar scoring head $r(x, y)$.

---

## 3. Split Design & Generalization Semantics

- **Train Split (In-Distribution):** 5 rules across 2 families (150 templates, 450 items, 150 CF pairs):
  - `renal_01` (metformin / glipizide, eGFR < 30)
  - `renal_03` (nitrofurantoin / fosfomycin, CrCl < 30)
  - `renal_04` (gabapentin dose reduction, CrCl 15–29)
  - `preg_01` (lisinopril / labetalol, pregnant == true)
  - `preg_03` (doxycycline / cefuroxime axetil, pregnant == true)
- **Near-OOD Held-Out (Seen trigger fact type, novel clinical intervention):** 2 rules (60 templates, 180 items, 60 CF pairs):
  - `renal_02` (enoxaparin dose reduction, CrCl < 30)
  - `preg_02` (methotrexate discontinuation / hydroxychloroquine, pregnant == true)
- **Far-OOD Held-Out (Completely unseen rule mechanism & fact type):** 3 rules (90 templates, 270 items, 90 CF pairs):
  - `allergy_01` (penicillin anaphylaxis / azithromycin for strep pharyngitis)
  - `allergy_02` (sulfonamide anaphylaxis/SJS / ciprofloxacin for cystitis)
  - `allergy_03` (aspirin AERD / acetaminophen for musculoskeletal pain)

---

## 4. Evaluation Metrics & Endpoints

### Hardware Noise Floor & Tie-Band Tolerance
- Dynamic padding invariance on RTX 5060 Laptop GPU establishes hardware floating point noise floor:  
  $\max \vert \text{score diff} \vert = 0.0625$.
- Pinned tie-band threshold:  
  $$\epsilon = 2 \times \max \vert \text{score diff} \vert = 0.1250$$
- Any preference margin with $\vert M \vert \le \epsilon$ is scored as a non-decisive tie.

### Metrics Defined
1. **Headline Metric: Discriminative Flip Rate (DFR):**  
   The fraction of scenario quartets where the base item, edited item, AND near-miss item are all decisively correct:
   $$\text{DFR} = \frac{1}{N} \sum_{s=1}^N \mathbf{1}_{\{M_{\text{base}} > \epsilon \;\land\; M_{\text{edited}} > \epsilon \;\land\; M_{\text{near\_miss}} > \epsilon\}}$$
   (We will also report DFR minus the null-edit flip rate to account for the noise floor.)
2. **Diagnostic Metric: Over-flip Rate:**  
   The fraction of scenarios where the model decisively overrides on the edited item but erroneously flips on the near-miss item:
   $$\text{Over-flip Rate} = \frac{1}{N} \sum_{s=1}^N \mathbf{1}_{\{M_{\text{edited}} > \epsilon \;\land\; M_{\text{near\_miss}} < -\epsilon\}}$$
3. **Noise Floor Metric: Null-Edit Flip Rate:**  
   The fraction of scenarios where the model changes its prediction simply due to paraphrasing of non-decisive fields.
4. **Secondary Metric: Counterfactual Flip Rate (CFR):**  
   The fraction of scenarios where both base and edited items are decisively correct:
   $$\text{CFR} = \frac{1}{N} \sum_{s=1}^N \mathbf{1}_{\{M_{\text{base}} > \epsilon \;\land\; M_{\text{edited}} > \epsilon\}}$$
5. **Pairwise Accuracy:**  
   Strict pairwise accuracy with tie band: fraction of items where $M = r(\text{chosen}) - r(\text{rejected}) > \epsilon$.

---

## 5. Non-Neural Baselines & Shortcut Gating Criteria

Every results table must report the following non-neural baselines alongside reward models:
1. **Always-Default:** Predicts default option for all items (pairwise accuracy 50.0% on balanced CF pairs, 66.7% on pooled triplets).
2. **Response-Only Logistic Regression:** Trained and tested on response text alone (verifies 50.0% exact chance, zero response leakage).
3. **Keyword Heuristic:** Predicts alternative if trigger keywords appear (achieves 100% on CF pairs, but over-flips on near-misses, collapsing DFR).
4. **Prompt-Only Bag-of-Words Logistic Regression:** Evaluates whether prompt n-grams predict the correct action.

### Preregistered Gating Rule
- Gating threshold applies **strictly to BALANCED accuracy on base-vs-edited counterfactual pairs**.
- If Far-OOD balanced CF accuracy exceeds **70.0%**, the dataset is flagged for lexical shortcut leakage and execution STOPS.
- Near-miss accuracy is reported separately.
- Raw pooled accuracy across triplets (66.7% default base rate) is reported for completeness, but is **never used for gating**.

---

## 6. Training Loss & Class Balance

To prevent default-dominance during post-training:
- **Loss Weighting:** Edited items (alternative chosen) receive sample weight **2.0**; base items and near-miss items receive sample weight **1.0**.
- **Effective Class Balance:** Total weight for default-chosen ($1.0 + 1.0 = 2.0$) equals total weight for alternative-chosen ($2.0$), guaranteeing a **50/50 effective class balance** in the training loss across all conditions.
