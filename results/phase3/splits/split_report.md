# Phase 3 — Leakage-Safe Grouped Dataset Split Report

Generated at: `2026-10-01T08:50:58.797193+00:00` · Git commit: `78baa1bec4a7bedf34076806fa83c7435ec75392` · Dirty: `True`
Input table: `data\processed\remedqa_pairs.parquet` (SHA-256: `25cc18e915260f6d…`)

## 1. Splitting Principles & Guarantees

- **Atomic Grouping:** All 7 conditions for a question (`mcq`, `roman_numeral`, `none_of_the_provided`, `fixed_pos`, `incorrect`, `open`, `no_symbols`) are strictly assigned to the exact same split.
- **Stem Collision Neutralization:** Questions sharing identical normalized question text (transitive equivalence classes via graph connected components) are grouped together into clusters, preventing question-stem leakage across splits.
- **Pilot & Phase 2A Cohort Holdout:** All 200 pilot questions and all 50 Phase 2A questions are guaranteed to reside in the **held-out test split** (0 in train, 0 in val).
- **Stratification:** Stratified by source question bank (`medqa`, `medmcqa`, `mmlu`) with target ratios ~70% train / ~10% val / ~20% test.

## 2. Split Summary

| Split | Questions (N) | % Questions | Total Pairs Rows | medqa | medmcqa | mmlu | Pilot 200 | Phase 2A (50) |
|---|---|---|---|---|---|---|---|---|
| **train** | 2207 | 70.0% | 13,242 | 881 | 700 | 626 | 0 | 0 |
| **val** | 316 | 10.0% | 1,896 | 126 | 100 | 90 | 0 | 0 |
| **test** | 631 | 20.0% | 3,786 | 252 | 200 | 179 | 200 | 50 |

**Totals:** 3154 questions across 3084 clusters, 18,924 pairing rows.

## 3. Leakage Verification Results

- **Question ID Disjointness:** PASSED (0 shared IDs across any pair of splits)
- **Normalized Question Text Disjointness:** PASSED (0 shared stems across any pair of splits)
- **Pilot Cohort Containment:** PASSED (200/200 pilot questions in test, 0 in train/val)
- **Phase 2A Cohort Containment:** PASSED (50/50 Phase 2A questions in test, 0 in train/val)
- **Representation Completeness:** PASSED (all questions have 6 perturbation rows + 1 mcq baseline = 7 conditions)
- **Overall Status:** **PASSED — ZERO LEAKAGE DETECTED**

## 4. Artifact Manifest

```
data/processed/phase3/
  ├── split_manifest.csv      # 3,154 rows: question_id, source, cluster, split, flags
  ├── train.parquet           # 13,242 rows (2,207 questions x 6 perturbations)
  ├── val.parquet             # 1,896 rows (316 questions x 6 perturbations)
  └── test.parquet            # 3,786 rows (631 questions x 6 perturbations)
results/phase3/splits/
  ├── split_report.json       # Structured configuration and provenance
  └── split_report.md         # This report
```