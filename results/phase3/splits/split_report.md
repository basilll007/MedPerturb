# Phase 3 — Four-Way Leakage-Safe Dataset Split Report

Generated at: `2026-10-01T08:59:19.707418+00:00` · Git commit: `86bcfe610f0b437669691f37c6b204878986d197` · Dirty: `True`
Input table: `data\processed\remedqa_pairs.parquet` (SHA-256: `25cc18e915260f6d…`)

## 1. Splitting Principles & Four-Way Architecture

- **Atomic Grouping:** All 7 conditions for a question (`mcq`, `roman_numeral`, `none_of_the_provided`, `fixed_pos`, `incorrect`, `open`, `no_symbols`) are strictly assigned to the exact same split.
- **Stem Collision Neutralization:** Questions sharing identical normalized question text (transitive equivalence classes via graph connected components) are grouped together into clusters, preventing question-stem leakage across splits.
- **Diagnostic vs. Final Test Separation:** The 200-question pilot cohort and 50-question Phase 2A subset are strictly isolated inside `diagnostic` ($N=207$ with stem-linked questions). The `final_test` split ($N=424$) contains **only previously unexamined questions** with zero pilot or Phase 2A overlap.
- **Stratification:** Stratified by source question bank (`medqa`, `medmcqa`, `mmlu`) across all splits.

## 2. Four-Way Split Summary

| Split | Questions (N) | % Questions | Total Pairs Rows | medqa | medmcqa | mmlu | Pilot (200) | Phase 2A (50) | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| **`train`** | 2207 | 70.0% | 13,242 | 881 | 700 | 626 | 0 | 0 | LoRA / GRPO post-training |
| **`val`** | 317 | 10.1% | 1,902 | 126 | 100 | 91 | 0 | 0 | Model selection & checkpointing |
| **`diagnostic`** | 207 | 6.6% | 1,242 | 80 | 63 | 64 | 200 | 50 | Inspected pilot & Phase 2A benchmark comparison |
| **`final_test`** | 423 | 13.4% | 2,538 | 172 | 137 | 114 | 0 | 0 | Untouched final held-out evaluation |

**Totals:** 3154 questions across 3084 clusters, 18,924 pairing rows.

## 3. Leakage Verification Results

- **Question ID Disjointness:** PASSED (0 shared IDs across any pair of the 4 splits)
- **Normalized Question Text Disjointness:** PASSED (0 shared stems across any pair of the 4 splits)
- **Pilot Cohort Isolation:** PASSED (200/200 pilot questions in diagnostic, exactly 0 in train, val, or final_test)
- **Phase 2A Cohort Isolation:** PASSED (50/50 Phase 2A questions in diagnostic, exactly 0 in train, val, or final_test)
- **Representation Completeness:** PASSED (all questions have 6 perturbation rows + 1 mcq baseline = 7 conditions)
- **Overall Status:** **PASSED — ZERO LEAKAGE DETECTED**

## 4. Artifact Manifest

```
data/processed/phase3/
  ├── split_manifest.csv      # 3,154 rows: question_id, source, cluster, split, flags
  ├── train.parquet           # 13,242 rows (2,207 questions x 6 perturbations)
  ├── val.parquet             # 1,896 rows (316 questions x 6 perturbations)
  ├── diagnostic.parquet      # 1,242 rows (207 questions x 6 perturbations)
  └── final_test.parquet      # 2,544 rows (424 questions x 6 perturbations)
results/phase3/splits/
  ├── split_manifest.csv      # Audit copy
  ├── split_report.json       # Structured configuration and provenance
  └── split_report.md         # This report
```