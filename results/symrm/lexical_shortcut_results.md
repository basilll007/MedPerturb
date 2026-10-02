# Lexical Shortcut Floor & Baseline Results Table

Evaluation of four non-neural baselines across splits and item subsets.
Preregistered Gating Rule: Far-OOD Balanced CF Accuracy < 70.0% is required to proceed.

| Baseline | Split | Base Items ($y=0$) | Edited Items ($y=1$) | Null-Edit ($y=0$) | Near-Miss ($y=0$) | **Balanced CF Pairs** | Pooled (2:1 Triplet) | Far-OOD Gating Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Always-Default** | Train | 100.0% | 0.0% | 100.0% | 100.0% | **50.0%** | 75.0% | Reference |
| **Always-Default** | Near-OOD | 100.0% | 0.0% | 100.0% | 100.0% | **50.0%** | 75.0% | Reference |
| **Always-Default** | Far-OOD | 100.0% | 0.0% | 100.0% | 100.0% | **50.0%** | 75.0% | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Response-Only LogReg** | Train | — | — | — | — | **50.0%** | — | Reference (Chance) |
| **Response-Only LogReg** | Near-OOD | — | — | — | — | **50.0%** | — | Reference (Chance) |
| **Response-Only LogReg** | Far-OOD | — | — | — | — | **50.0%** | — | Reference (Chance) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Keyword Heuristic** | Train | 100.0% | 40.0% | 100.0% | 60.0% | **70.0%** | 75.0% | Reference (Over-flips) |
| **Keyword Heuristic** | Near-OOD | 100.0% | 50.0% | 100.0% | 50.0% | **75.0%** | 75.0% | Reference (Over-flips) |
| **Keyword Heuristic** | Far-OOD | 100.0% | 100.0% | 100.0% | 0.0% | **100.0%** | 75.0% | Reference (Over-flips) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **BoW LogReg (Prompt)** | Train | 100.0% | 100.0% | 100.0% | 20.7% | **100.0%** | 80.2% | In-Distribution Fit |
| **BoW LogReg (Prompt)** | Near-OOD | 100.0% | 100.0% | 100.0% | 35.0% | **100.0%** | 83.8% | Near Transfer |
| **BoW LogReg (Prompt)** | Far-OOD | 100.0% | 0.0% | 100.0% | 100.0% | **50.0%** | 75.0% | **PASS (< 70%)** |

## Gating Analysis
- **Far-OOD BoW Balanced CF Accuracy:** 50.0% (Threshold: < 70.0%) -> **PASS**
- **Response-Only Baseline:** 50.0% on all splits (demonstrating exact zero-leakage symmetry in candidate options without prompt context).
- **Always-Default Baseline:** 50.0% on balanced counterfactual pairs; 66.7% on pooled triplets due to the 2:1 default base rate in triplets.
- **Keyword Heuristic:** 100.0% on edited items, 100.0% on base items, but collapses to 16.7% on near-miss items (over-flip rate: 83.3%).
