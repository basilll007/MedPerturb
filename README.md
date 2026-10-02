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
| Phase 3 — Post-Training & Untouched Final Evaluation | **Complete** — Unsloth GRPO post-training, 4-way leakage-safe split, 8,883 untouched final test evaluations, paired McNemar/Bootstrap tests, figures, and tables. See `docs/phase3.html`. |
| **SymRM Pilot (Gate 2a Resubmission)** | **Awaiting PI review** — 900 vignettes, 0 banned tokens, raw-value diff invariance, 21/21 QC tests passed, Far-OOD BoW floor 50.0% (< 70% threshold). Data tagged `v0-unreviewed`. See `docs/symrm.html`. |

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

## Reproducing Phase 3 (Post-Training & Untouched Evaluation)

Phase 3 tests whether symbolic-consistency reward during post-training reduces required adaptation failures under task transformations while preserving semantic consistency under meaning-preserving transformations.

### 1. Zero-Leakage 4-Way Splitting
Constructs train, validation, diagnostic (containing all pilot/Phase 2A questions), and untouched final_test splits using connected-component clustering across normalized question stems:
```powershell
uv run --no-sync python scripts\phase3_split.py
uv run --no-sync pytest tests\test_phase3_split.py
```

### 2. Unit Testing Reward Functions
Validates both task correctness ($R_{\text{correct}}$) and symbolic constraint satisfaction ($R_{\text{symbolic}}$):
```powershell
uv run --no-sync pytest tests\test_symbolic_rewards.py tests\test_paired_stats.py
```

### 3. Post-Training via Unsloth GRPO (8 GB VRAM Budget)
Trains Policy A (Correctness-Only) and Policy B (Neuro-Symbolic) with identical base checkpoint (`unsloth/Qwen3-4B-unsloth-bnb-4bit`), LoRA rank 16, 60 optimizer steps, seed 42:
```powershell
# Policy A: Correctness-Only GRPO
uv run --no-sync python scripts\phase3_train_grpo.py --reward_mode correctness --output_dir results\phase3\checkpoints\policy_correctness

# Policy B: Neuro-Symbolic GRPO (R_correct + 1.0 * R_symbolic)
uv run --no-sync python scripts\phase3_train_grpo.py --reward_mode neurosymbolic --output_dir results\phase3\checkpoints\policy_neurosymbolic
```

### 4. Master Final Evaluation, Hypothesis Tests, Figures & Tables
Executes the untouched final test evaluation (423 questions x 7 conditions = 2,961 calls per model; 8,883 calls total), runs paired McNemar tests and bootstrap CIs, generates error analysis reports, 300 DPI figures, and manuscript tables:
```powershell
uv run --no-sync python scripts\phase3_run_final_evaluations.py --stage final_test
```

### Key Experimental Findings (Untouched Final Test, N = 423 Questions)
| Model Policy | MCQ Acc | Invariance Acc | ReAcc (Joint Invariance) | ReCon (Consistency) | Adaptation Failure Rate [Primary] |
|---|---|---|---|---|---|
| Base Qwen3-4B (4-bit) | 56.50% | 51.38% | 34.75% | 50.59% | 72.93% |
| Policy A: Correctness GRPO | 56.50% | 51.85% | 34.75% | 50.59% | 73.29% |
| Policy B: Neuro-Symbolic GRPO | 56.26% | **52.09%** | **35.46%** (+0.71%) | **51.78%** (+1.19%) | 73.52% |

- **Primary Hypothesis Test (Adaptation Failure)**: Exact McNemar test $p = 0.8238$ (discordant: 11 vs 9), 95% Bootstrap CI: [-0.0083, +0.0130]. Null hypothesis retained ($p \ge 0.05$). Under a 60-step LoRA regime on 4-bit Qwen3-4B, symbolic-consistency reward does not significantly reduce adaptation failure compared to correctness-only reward.
- **Secondary Gains (Invariance Robustness)**: Neuro-symbolic training achieved slight but statistically non-significant improvements in invariance, with joint correctness (ReAcc: 35.46%, +0.71%), prediction consistency (ReCon: 51.78%, +1.19%), position-shift accuracy (52.72%), and lower representation instability (204 vs 215).

## SymRM: Symbolic Supervision for Neural Reward Models (NAACL 2027 Main Track)

### Research Question & Hypothesis
*"Can symbolic supervision teach neural reward models to generalize beyond the rules they were trained on?"*

Neural reward models (RMs) cover arbitrary inputs but frequently fail to let decisive patient evidence override default heuristics ("knows but does not use"). Symbolic verifiers provide exact constraint satisfaction, but only on rules manually authored by humans. We test whether a symbolic verifier, used strictly during post-training, forces neural RMs to adapt correctly to decisive clinical evidence on held-out rules where the verifier is absent at inference.

**Interactive Dashboard:** Open [`docs/symrm.html`](docs/symrm.html) in any browser for interactive figures, full scenario triplet inspector, demographic distributions, and baseline breakdowns.

### Gate Progression & Current Status

| Gate Stage | Objective | Status | Artifacts & Evidence |
| :--- | :--- | :--- | :--- |
| **Gate 0** | Hardware verification, VRAM sweep, padding invariance | **PASSED** | Peak VRAM 3.78 GB at BS=4 (GA=8), padding margin sign pass, `results/symrm/compute_log.csv` |
| **Gate 1** | Response option neutrality & near-miss redesign | **Conditionally approved (pending pharmacist review)** | Plain prescription syntax, zero rationale, mean token \|diff\| = 0.90 &le; 2.0, `configs/symrm/rules.yaml` |
| **Gate 2a** | Answer leakage elimination & prompt deduplication | **Awaiting PI review** | Raw values only, 0 banned tokens, field diff invariance, Jaccard max 0.7849 &le; 0.90, BoW floor 50.0% (PASS) |
| **Gate 2b** | Human clinical sign-off & pharmacist review | Pending | All rules flagged `verified: false`, dataset tagged `v0-unreviewed` |
| **Gate 3** | Probe training & RM evaluation across splits | Not yet selected | Trained open-weight RMs on RTX 5060 (8GB) |

### Key Figures & Visualizations

| Figure | Description | File Path |
| :--- | :--- | :--- |
| **Figure 1 (QC Diagnostic)** | Hardware Scaling & VRAM Sweep on RTX 5060 (8GB) | [`figures/symrm/fig1_vram_and_throughput_scaling.png`](figures/symrm/fig1_vram_and_throughput_scaling.png) |
| **Figure 2** | Lexical Shortcut Floors Across Splits | [`figures/symrm/fig2_lexical_shortcut_floors.png`](figures/symrm/fig2_lexical_shortcut_floors.png) |
| **Figure 3** | Clinical Family Breakdown & Demographics (Age 25–80, Sex) | [`figures/symrm/fig3_dataset_composition_and_demographics.png`](figures/symrm/fig3_dataset_composition_and_demographics.png) |
| **Figure 4 (QC Diagnostic)** | Prompt Deduplication & Pairwise Jaccard Distribution | [`figures/symrm/fig4_template_diversity_and_deduplication.png`](figures/symrm/fig4_template_diversity_and_deduplication.png) |
| **Figure 5 (QC Diagnostic)** | Response Option Symmetry Audit (Mean \|Diff\| &le; 2.0) | [`figures/symrm/fig5_token_length_neutrality.png`](figures/symrm/fig5_token_length_neutrality.png) |

### Lexical Shortcut Floors & Baseline Evaluation

Evaluation of four non-neural baselines across splits and item subsets.
*Preregistered Gating Rule: Far-OOD Balanced CF Accuracy < 70.0% is required to proceed.*

| Baseline | Split | Base Items ($y=0$) | Edited Items ($y=1$) | Near-Miss Items ($y=0$) | **Balanced CF Pairs** | Pooled (2:1 Triplet) | Far-OOD Gating Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Always-Default** | Train | 100.0% | 0.0% | 100.0% | **50.0%** | 66.7% | Reference |
| **Always-Default** | Near-OOD | 100.0% | 0.0% | 100.0% | **50.0%** | 66.7% | Reference |
| **Always-Default** | Far-OOD | 100.0% | 0.0% | 100.0% | **50.0%** | 66.7% | Reference |
| **Response-Only LogReg** | All Splits | — | — | — | **50.0%** | — | Reference (Chance) |
| **Keyword Heuristic** | Train | 100.0% | 40.0% | 60.0% | **70.0%** | 66.7% | Over-flips on near-miss |
| **Keyword Heuristic** | Near-OOD | 100.0% | 50.0% | 50.0% | **75.0%** | 66.7% | Over-flips on near-miss |
| **Keyword Heuristic** | Far-OOD | 100.0% | 100.0% | 0.0% | **100.0%** | 66.7% | 100% near-miss over-flip |
| **BoW LogReg (Prompt)** | Train | 100.0% | 100.0% | 20.7% | **100.0%** | 73.6% | In-Distribution Fit |
| **BoW LogReg (Prompt)** | Near-OOD | 100.0% | 100.0% | 35.0% | **100.0%** | 78.3% | Near Transfer |
| **BoW LogReg (Prompt)** | **Far-OOD** | **100.0%** | **0.0%** | **100.0%** | **50.0%** | **66.7%** | **PASS (< 70.0%)** |

### Reproducing SymRM Pilot Artifacts

```powershell
# 1. Regenerate 900-vignette counterfactual dataset (tagged v0-unreviewed)
uv run python src/symrm/data/generator.py

# 2. Run the 21-test automated QC suite (Neutrality, Diff Invariance, Polarity, Padding)
uv run pytest -v tests/symrm

# 3. Evaluate non-neural shortcut baselines & update results table
uv run python src/symrm/eval/lexical_shortcuts.py

# 4. Run template diversity & deduplication audit
uv run python src/symrm/data/diversity.py

# 5. Generate high-resolution figures for dashboard and publication
uv run python scripts/generate_symrm_figures.py
```

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
