# Medical LLM Reliability Under Perturbation — NAACL 2027

Empirical study of whether medical-domain LLMs give reliable answers across
meaning-preserving and format-altering reformulations of the same
multiple-choice question — and whether a model shows measurable evidence of
its own unreliability when it fails under such a perturbation.

**Central question:** *When a medical LLM fails under a meaning-preserving
perturbation, does the model provide measurable evidence that its prediction
has become unreliable?*

This is a falsification-first project (FAILURE → MEASUREMENT → EXPLANATION →
INTERVENTION), not an attempt to chase state-of-the-art MedQA accuracy or to
prove the hypothesis regardless of evidence.

**Dashboard:** open `docs/index.html` in a browser for the full interactive
write-up (dataset forensics, perturbation semantics, pairing methodology,
descriptive charts, pilot cohort, Phase 1A smoke-test status). The `docs/`
folder is GitHub-Pages-ready if this repo is pushed to GitHub.

## Status

| Phase | Status |
|---|---|
| Phase 0 — dataset forensics | **Complete** — see `results/audit/dataset_audit.md`, classification: **YELLOW** |
| Phase 1A — controlled behavioral pilot (smoke test) | **Partial, stopped** — Gemini smoke test complete (20/20); Claude smoke test blocked by an Anthropic account usage cap (resets 2026-10-01 00:00 UTC). Remaining 190-question cohort **not run**, pending approval. See `results/audit/PHASE1A_SMOKE_TEST_REPORT.md`. |
| Phase 1B+ / representation analysis | Not started — deferred, requires GPU + open-weight model |

Key finding from Phase 0 worth knowing before reading anything else: only 2
of ReMedQA's 7 perturbation types (`roman_numeral`, and originally
`fixed_pos`) leave the question genuinely meaning-preserving, and `fixed_pos`
turned out to have its own confound (the correct answer is placed at
position D 100% of the time — confirmed from the ReMedQA paper's own design
notes, not just local diffs). See `docs/perturbations.html` for the full
breakdown.

## Setup

```powershell
# Python 3.11, dependency/env management via uv
uv sync

# API credentials for Phase 1A (Claude + Gemini) — create .env in project root:
#   ANTHROPIC_API_KEY=...
#   GEMINI_API_KEY=...
# .env is gitignored; never commit it.
```

All commands below are run from the project root in PowerShell:
`uv run python scripts\<name>.py`

## Repository structure

```
data/
  raw/            Downloaded datasets (ReMedQA, CareQA-en, MedQA) — gitignored, untouched by any script
  processed/      remedqa_pairs.parquet, pilot_200.parquet + method docs — gitignored, regenerate via scripts/
scripts/
  _audit_*.py             Per-dataset forensic audits (Phase 0, Tasks 2-4)
  audit_datasets.py        Cross-dataset overlap analysis + dataset_manifest.csv (Tasks 5-7)
  build_pairs.py           Original -> perturbed pairing table (Task 8)
  descriptive_analysis.py  Perturbation descriptive stats + figures (Task 9)
  build_pilot.py           Deterministic 200-question pilot cohort, seed=42 (Task 10)
  phase1a_prompts.py       No-CoT prompt templates for Phase 1A
  phase1a_parse.py         Answer parsing, kept separate from inference
  phase1a_run_smoke.py     Phase 1A smoke test runner (cached, restart-safe, capped at 40 calls)
results/
  audit/          Dataset audit report, manifest, perturbation descriptives, Phase 1A smoke test report
  behavioral/     raw/ (per-call API responses, gitignored) and parsed/ (scored CSVs) from Phase 1A
figures/audit/    4 descriptive-analysis PNGs (also embedded in docs/)
logs/             Structured JSONL run logs (gitignored)
docs/             Interactive dashboard (5 pages) — open docs/index.html
notebooks/        Exploratory analysis (none committed yet)
src/download_data.py   Original dataset download script (HF -> data/raw/)
```

## Datasets

| Dataset | Path | Role | Independence |
|---|---|---|---|
| ReMedQA | `data/raw/remedqa` | Primary — 3 source banks (MedQA/MedMCQA/MMLU) x 7 perturbations, id-paired | — |
| CareQA (en) | `data/raw/careqa_en` | External validation | **Verified independent** — 0% overlap with ReMedQA and MedQA |
| MedQA (local) | `data/raw/medqa` | Intended control | **Not independent of ReMedQA** — local `test` split is 98.9%/100% duplicated with ReMedQA's `medqa_mcq` family. Use `train`/`validation` (0% overlap) instead. |

Full detail: `results/audit/dataset_audit.md`.

## Reproducing Phase 0

```powershell
uv run python scripts\audit_datasets.py        # dataset audit, overlap analysis, manifest
uv run python scripts\build_pairs.py            # original -> perturbed pairing table (18,924 rows)
uv run python scripts\descriptive_analysis.py   # perturbation descriptives + 4 figures
uv run python scripts\build_pilot.py            # deterministic 200-question pilot cohort (seed=42)
```

## Reproducing the Phase 1A smoke test

Requires `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in `.env`. Makes at most 40
real API calls (10 questions x 2 conditions x 2 models); already-cached
responses under `results/behavioral/raw/` are skipped, so a rerun after an
interruption or a rate-limit block does not re-pay for completed requests.

```powershell
uv run python scripts\phase1a_run_smoke.py
```

This script deliberately does **not** accept a flag to run the full
200-question cohort — scaling past the smoke test is a separate, explicitly
approved step.

## Research guardrails

- No dataset field, split correspondence, or metric definition is assumed —
  everything is verified by running code against the on-disk data.
- Perturbations are never pooled; meaning-preserving and task-altering
  perturbations are scored and reported separately.
- Accuracy loss under perturbation is not, by itself, treated as evidence of
  "hallucination" or "lack of reasoning" — see `results/audit/behavioral_experiment_plan.md`
  for the statistical plan (McNemar's test for paired correctness, cluster-level
  bootstrap CIs, no unpaired t-tests on paired binary outcomes).
- Raw datasets are never modified in place; all derived artifacts live under
  `data/processed/`, `results/`, or `figures/`.
