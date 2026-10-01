# Template Diversity & Lexical Similarity Report

Analysis of sentence skeleton variation, pairwise prompt Jaccard similarity per rule, and cross-scenario deduplication.
Quality Constraints:
1. FLAG any rule with mean pairwise Jaccard similarity > 0.80.
2. Prompt Deduplication: No two prompts from DIFFERENT scenarios with Jaccard > 0.90.

## 1. Per-Rule Diversity Table

| Rule ID | Family | Split | Num Items | Unique Skeletons | Mean Jaccard | Min Jaccard | Max Jaccard (Triplets) | Status (Jaccard $\le 0.80$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `renal_01` | renal | `train` | 90 | 30 | **0.547** | 0.407 | 0.975 | PASS |
| `renal_03` | renal | `train` | 90 | 30 | **0.518** | 0.386 | 0.974 | PASS |
| `renal_04` | renal | `train` | 90 | 30 | **0.500** | 0.359 | 0.973 | PASS |
| `preg_01` | pregnancy | `train` | 90 | 90 | **0.538** | 0.398 | 0.944 | PASS |
| `preg_03` | pregnancy | `train` | 90 | 90 | **0.516** | 0.359 | 0.953 | PASS |
| `renal_02` | renal | `near_ood` | 90 | 30 | **0.540** | 0.379 | 0.956 | PASS |
| `preg_02` | pregnancy | `near_ood` | 90 | 90 | **0.522** | 0.363 | 0.955 | PASS |
| `allergy_01` | allergy | `far_ood` | 90 | 90 | **0.520** | 0.355 | 0.966 | PASS |
| `allergy_02` | allergy | `far_ood` | 90 | 90 | **0.513** | 0.352 | 0.966 | PASS |
| `allergy_03` | allergy | `far_ood` | 90 | 90 | **0.527** | 0.366 | 0.955 | PASS |

## 2. Cross-Scenario Deduplication (Distinct Scenarios)
- **Max Pairwise Jaccard Across Different Scenarios:** **0.7849** (Threshold: $\le 0.90$) -> **PASS**
- **Highest Similarity Scenario Pair:** tpl_preg_03_013 vs tpl_preg_03_019
- **Mean Pairwise Jaccard Across Different Scenarios:** 0.4324

## 3. Diversity Summary
**STATUS: PASS (All rules mean Jaccard $\le 0.80$; All cross-scenario pairs Jaccard $\le 0.90$)**
Clinical templates exhibit strong demographic, presentation, and distractor variability.
