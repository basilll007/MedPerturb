# Phase 0 Dataset Audit — Consolidated Report

Project: NAACL 2027 medical-LLM-reliability paper. This report consolidates
the forensic audits of the three raw datasets (`_remedqa_findings.md`,
`_careqa_findings.md`, `_medqa_findings.md`, produced by
`scripts/_audit_{remedqa,careqa,medqa}.py`) plus the Task 5 cross-dataset
overlap analysis and Task 6 manifest, all reproduced end-to-end by
`scripts/audit_datasets.py`. Every number in this report comes from running
code against the on-disk data at `data/raw/{remedqa,careqa_en,medqa}` — none
is hand-derived or estimated. No `data/processed` or `figures/` files were
read or modified while producing this report.

Reproduce with: `uv run python scripts\audit_datasets.py` (from project root).

---

## 1. Dataset Structure

| Dataset | On-disk path | HF container | Splits |
|---|---|---|---|
| ReMedQA | `data/raw/remedqa` | `DatasetDict` | 21 splits, named `{source}_{perturbation}`, `source` ∈ {medqa, medmcqa, mmlu} × `perturbation` ∈ {mcq, open, incorrect, roman_numeral, none_of_the_provided, fixed_pos, no_symbols} |
| CareQA (en) | `data/raw/careqa_en` | `DatasetDict` | 1 split: `test` |
| MedQA (local) | `data/raw/medqa` | `DatasetDict` | 3 splits: `train`, `validation`, `test` |

ReMedQA is a *derived/perturbed* dataset built from three source question
banks (MedQA, MedMCQA, MMLU); it is not itself a raw source corpus. Local
MedQA and CareQA are independently-downloaded raw corpora used to test
whether ReMedQA's `medqa_*` family and MedMCQA/MMLU-style content leak into
data a researcher might otherwise treat as "held-out."

## 2. Exact Schemas

**ReMedQA** (identical schema across all 21 splits):
`id:string, question:string, options:string, answer:string, prompt:string, prompt_think:string`.
`options` and `answer` are **Python-repr strings**, not JSON — parse with
`ast.literal_eval`, not `json.loads` (verified: `json.loads` silently
misparses some numeric-looking `answer` values, e.g. MedMCQA answers like
`"0.01"` parse as a JSON float rather than the intended string/option key).

**CareQA (en)**, single `test` split:
`exam_id:int64, question:string, op1:string, op2:string, op3:string, op4:string, cop:int64, year:int64, category:string, unique_id:string`.
`cop` ∈ {1,2,3,4}, a 1-indexed pointer into `op1..op4` (inferred from data
layout, not an HF card assertion). No column literally named `id`;
`unique_id` is unique across all 5621 rows and is used as CareQA's row
identifier here. `exam_id` is **not** unique (210 distinct values over 5621
rows — multiple questions share an exam).

**MedQA (local)**, three splits (train/validation/test), identical schema:
`question:string, answer:string, options:{A:string,B:string,C:string,D:string}, meta_info:string, answer_idx:string, metamap_phrases:list[string]`.
**No id/uid column of any kind** — row position is the only handle.
`answer` (free text) was cross-checked against `options[answer_idx]` on a
200-row sample per split: 200/200 match in all three splits.

## 3. Sample Counts

| Dataset | Split | n |
|---|---|---|
| ReMedQA | medqa_* (×7) | 1259 each (8813 total) |
| ReMedQA | medmcqa_* (×7) | 1000 each (7000 total) |
| ReMedQA | mmlu_* (×7) | 895 each (6265 total) |
| ReMedQA | **all 21 splits** | **22078 total rows, 3154 unique (id, source) questions** |
| CareQA | test | **5621** |
| MedQA (local) | train | **10178** |
| MedQA (local) | validation | **1272** |
| MedQA (local) | test | **1273** |
| MedQA (local) | **all 3 splits** | **12723 total** |

## 4. ReMedQA Perturbation Organization

All 7 perturbations are applied per-source (medqa/medmcqa/mmlu) to the same
underlying question, joined by a shared `id`. Direct before/after inspection
against the `mcq` baseline (method: for a sample `id` shared across all 7
perturbations, diff `options`/`answer` for that id across perturbations —
see `_remedqa_findings.md` §8) shows:

| Perturbation | What it actually changes | Meaning-preserving (of the QUESTION)? |
|---|---|---|
| `mcq` | Baseline: unperturbed question, 4 lettered options A–D, single-letter answer. | Baseline (n/a) |
| `roman_numeral` | **Only** the option labels change, A/B/C/D → I/II/III/IV. Question text, option text, and correct-answer content are byte-identical to `mcq`. | **Yes** — clean, meaning-preserving option-*label* format perturbation. |
| `fixed_pos` | Option **order** is shuffled relative to `mcq`; the answer letter changes to track the shuffled position, but option text, question text, and answer content are unchanged. | **Yes** — clean, meaning-preserving option-*order* format perturbation. |
| `open` | Question text is **reworded** into an open-ended, non-MCQ phrasing (e.g. "Which of the following is the correct next action..." → "What is the correct next action..."), options are dropped from the prompt, and `answer` becomes free text instead of a letter. | **No** — this rewords the question and switches the task from MCQ selection to free-form generation. Not meaning-preserving of the *task format*. |
| `no_symbols` | Verified directly: **question text is unchanged** from `mcq` (not reworded — this contradicts a plausible assumption that it, like `open`, rewords the question). Options are still shown but without A/B/C/D labels; the expected `answer` becomes the full option text rather than a letter. | Question itself: yes, unchanged. Answer-output format: no — still a task-format change (label-based vs. text-based answer extraction), just not a question rewording. Treat as a distinct, narrower perturbation than `open`. |
| `incorrect` | Question and option text are unchanged, but the **task is inverted**: the model must identify the **three incorrect** options (`answer` becomes a 3-element list of letters) instead of the one correct option. | **No** — task-altering (select-incorrect vs. select-correct), not a meaning-preserving perturbation of the question. |
| `none_of_the_provided` | The **correct option's text is overwritten** with the literal string `"None of the provided options"`, and that overwritten option remains labeled as the correct answer. The actual correct-answer content is removed from the choice set entirely. | **No** — this destroys the correct answer's semantic content rather than preserving meaning; it changes what "correct" means for the item. |

**Bottom line for experimental design:** only `roman_numeral` and `fixed_pos`
are genuinely meaning-preserving format perturbations suitable for directly
testing a "meaning-preserving perturbation should not change model behavior"
hypothesis. `open`, `no_symbols`, `incorrect`, and `none_of_the_provided`
each alter the task in a substantive way (output format, task definition, or
available answer content) and should be analyzed/reported as a *separate*
category — "task-format" or "task-semantics" perturbations — not conflated
with the meaning-preserving pair.

## 5. Pairing Mechanism

Pairing across the 7 perturbations of a source family is **id-based**, and
was verified (not assumed) in `_remedqa_findings.md` §2–§3:

- No duplicate `id` values within any of the 21 splits (checked directly).
- Per source family, across all 7 perturbation splits: **union of ids =
  intersection of ids** — i.e. every id present in any perturbation split is
  present in *all seven*. Exact counts: medqa 1259/1259, medmcqa 1000/1000,
  mmlu 895/895 (union = intersection in all three cases; 0 ids missing from
  any perturbation; 0 ids appearing in only some-but-not-all perturbations).
- Index-based (positional) joins do **not** reliably reproduce this pairing
  for at least one source (see `_remedqa_findings.md` §5) — id-based joins
  are required; positional joins would silently mispair rows for sources
  where row order is not preserved across splits.

Conclusion: id-based pairing across all 7 perturbations within a source is
fully reliable and should be the only join method used downstream.

## 6. Data Quality Problems Found

- **ReMedQA**: 0 missing/null question, option, or answer fields across all
  21 splits (verified, `_remedqa_findings.md` §6). Within-split normalized-
  duplicate questions: `medqa_*` splits 0/1259; `medmcqa_*` splits 4–6 rows
  (out of 1000, e.g. `medmcqa_mcq` 998 distinct / 1000 rows, 4 duplicate
  rows); `mmlu_*` splits 136 rows out of 895 (only 827 distinct normalized
  questions in `mmlu_mcq`) — MMLU's medical subset re-uses stems across
  related MMLU categories more than MedQA/MedMCQA do. `answer`/`options`
  parsing pitfall: use `ast.literal_eval`, not `json.loads` (see §2 above).
- **CareQA**: 0 missing/null values across question/options/`cop` (verified,
  `_careqa_findings.md`). 5621 rows, 5569 distinct normalized questions: 38
  duplicate-question *groups* covering 90 rows total. Manual inspection of
  two of these groups (e.g. "Duchenne Muscular Dystrophy:") confirms they
  are **not** true duplicates — same short question stem, but different
  `exam_id`, different options, and different `cop` (i.e. same generic prompt
  reused across different exam administrations with different answer
  choices). Do not naively collapse these as duplicates. `exam_id` is not a
  usable row identifier (210 distinct values over 5621 rows); `unique_id` is.
- **MedQA (local)**: 0 missing/null values across all checked columns in all
  3 splits (verified, `_medqa_findings.md` §6). Within-split duplicates:
  `train` has 2 duplicate normalized-question groups (4 rows of 10178);
  `validation` and `test` have 0. Cross-split (train/validation/test)
  leakage: **0** normalized-question overlaps in any of the 3 pairwise
  comparisons (verified directly, not assumed) — the standard MedQA
  train/val/test split is clean of exact-text leakage. No id column exists
  at all, so any downstream joining against this dataset must be
  content-based (as Task 5 does) or positional only within this dataset.

## 7. Cross-Dataset Overlap (Task 5)

**Normalization** (defined once in `scripts/audit_datasets.py::normalize_question`,
reused for every comparison below, and matching the normalization already
used in `_audit_careqa.py`/`_audit_medqa.py`): lowercase → strip leading/
trailing whitespace → remove ASCII punctuation (`string.punctuation` via
`str.translate`) → collapse internal whitespace runs to a single space.

**Primary method:** exact match on normalized question text. Comparison
pools per the task spec: ReMedQA's `*_mcq` splits (baseline, unperturbed
question text) vs. local MedQA (train+val+test, and broken out per split)
and vs. CareQA; local MedQA vs. CareQA.

| Pair | n(A) | n(B) | distinct normalized shared | rows in A matched (%) | rows in B matched (%) |
|---|---|---|---|---|---|
| ReMedQA(medqa_mcq) ↔ MedQA(train) | 1259 | 10178 | 0 | 0 / 1259 (0.00%) | 0 / 10178 (0.00%) |
| ReMedQA(medqa_mcq) ↔ MedQA(validation) | 1259 | 1272 | 0 | 0 / 1259 (0.00%) | 0 / 1272 (0.00%) |
| ReMedQA(medqa_mcq) ↔ MedQA(test) | 1259 | 1273 | **1259** | **1259 / 1259 (100.00%)** | 1259 / 1273 (98.90%) |
| ReMedQA(medqa_mcq) ↔ MedQA(train+val+test) | 1259 | 12723 | 1259 | 1259 / 1259 (100.00%) | 1259 / 12723 (9.90%) |
| ReMedQA(medmcqa_mcq) ↔ CareQA(test) | 1000 | 5621 | 0 | 0 / 1000 (0.00%) | 0 / 5621 (0.00%) |
| ReMedQA(mmlu_mcq) ↔ CareQA(test) | 895 | 5621 | 0 | 0 / 895 (0.00%) | 0 / 5621 (0.00%) |
| ReMedQA(medqa_mcq) ↔ CareQA(test) | 1259 | 5621 | 0 | 0 / 1259 (0.00%) | 0 / 5621 (0.00%) |
| MedQA(train+val+test) ↔ CareQA(test) | 12723 | 5621 | 0 | 0 / 12723 (0.00%) | 0 / 5621 (0.00%) |

**Concrete matched examples** (ReMedQA `medqa_mcq` vs. local MedQA `test`,
verbatim question text, truncated to 200 chars):

1. `[ReMedQA] A 20-year-old man comes to the physician because of worsening gait unsteadiness and bilateral hearing loss for 1 month. He has had intermittent tingling sensations on both cheeks over this time period...`
   `[MedQA test] ` — identical text.
2. `[ReMedQA] A 52-year-old man comes to the physician because of generalized pruritus and raised, erythematous plaques on the skin over his hands, chest, and legs for 6 hours. He reports having clear liquid discha...`
   `[MedQA test] ` — identical text.
3. `[ReMedQA] A 59-year-old man presents to his primary care provider with fatigue, a progressively worsening cough with flecks of blood, shortness of breath, and dark urine. He reports feeling ill for the past 3 w...`
   `[MedQA test] ` — identical text.

These are exact, byte-for-byte-after-normalization matches on long,
highly-specific clinical vignette text (not generic short stems), so this is
not a normalization false-positive or a coincidental medical-exam-question
convention — it is genuine shared provenance.

**Secondary/optional near-duplicate scan** (token-set Jaccard similarity ≥
0.85 on normalized-text word sets, restricted to pairs sharing at least one
"rare" token — word length ≥5 chars with document frequency ≤8 on the
smaller side, for tractability; explicitly excludes anything already an
exact match above): **0 near-duplicate candidates found in any of the 5
pairs**, including MedQA(train+val+test) ↔ CareQA(test) (623,846 candidate
pairs evaluated via blocking, untruncated) and ReMedQA(medqa_mcq) ↔
CareQA(test) (62,737 pairs evaluated). This is a non-exhaustive, blocked scan
(a pair sharing no rare token is never compared), but the 0 result across
all pairs corroborates the exact-match finding rather than being an artifact
of a stricter method.

**Adversarial check — could this be a normalization artifact or generic
medical-exam-question convention?** No, for two reasons: (1) the matched
examples above are long (>150 char), highly specific clinical vignettes with
unique patient-presentation details (specific ages, symptom combinations,
timelines) — the kind of text that would not coincidentally recur across
independently-authored exam banks; a coincidence hypothesis is not
plausible for text this specific. (2) The overlap is *structured*, not
diffuse: it is 100% concentrated in MedQA's `test` split and touches 0% of
`train`/`validation`, which is exactly the pattern expected if ReMedQA's
`medqa_*` family was built by sampling from (a near-superset of) MedQA's
`test` split specifically, rather than a random pattern expected from
generic phrasing convention.

**Findings, stated explicitly:**

- **Is local MedQA the same underlying source as ReMedQA's `medqa_*`
  family?** Partially, and precisely characterizable: ReMedQA's `medqa_mcq`
  (1259 questions) is a **100% subset of local MedQA's `test` split** (1259
  of 1273 test questions matched exactly; the other 14 test questions were
  not selected into ReMedQA). ReMedQA's `medqa_*` family shares **0%**
  overlap with local MedQA's `train` (10178) or `validation` (1272) splits.
  So: local MedQA `test` is **not independent** of ReMedQA's `medqa_*`
  family (it is very nearly the same data) — using both as if they were
  independent evaluation sets would double-count almost the same 1259
  questions. Local MedQA `train`/`validation`, however, **are** additional,
  non-overlapping MedQA-sourced data usable for e.g. building a larger or
  different MedQA-based cohort.
- **Is CareQA independent of ReMedQA and MedQA?** Yes, based on the evidence:
  **0% exact-match overlap** and **0 near-duplicate candidates** against
  ReMedQA's `medqa_mcq`, `medmcqa_mcq`, and `mmlu_mcq` pools, and against
  local MedQA (all 3 splits combined). CareQA can be treated as an
  independent dataset for validation purposes with respect to both ReMedQA
  and MedQA, at least under normalized-exact-match and the near-duplicate
  scan's Jaccard≥0.85 threshold — this does not rule out topical/clinical
  overlap (both are medical exam question banks and will naturally cover
  overlapping medical knowledge), only textual duplication.

## 8. Implications for Experimental Design

- **Meaning-preserving perturbation hypothesis:** only `roman_numeral` and
  `fixed_pos` are clean, genuinely meaning-preserving format perturbations
  of the question (see §4) and should be the primary vehicle for testing
  whether the model's answer is invariant to meaning-preserving perturbation.
  `open`, `no_symbols`, `incorrect`, and `none_of_the_provided` each change
  the task in a substantive way and should be reported/analyzed as a
  *separate* category (task-format or task-semantics robustness), not
  pooled with `roman_numeral`/`fixed_pos` results, or the paper's central
  claim will be conflating two different phenomena (format sensitivity vs.
  task-definition sensitivity).
- **Local MedQA test split cannot be used as an independent replication of
  ReMedQA's `medqa_*` family** — it is essentially the same 1259 questions
  (100%/98.9% match). Any experimental design that evaluates on both and
  treats them as independent evidence will be double-counting. If a larger
  or fresh MedQA-based cohort is desired, local MedQA `train`
  (10178 rows, 0% overlap with ReMedQA) is the correct source to draw from,
  not `test`.
- **CareQA is a legitimate independent validation set** relative to both
  ReMedQA and local MedQA, per the 0%/0-near-duplicate result in §7. It
  provides genuinely held-out medical MCQ data (Spanish MIR-exam-derived,
  English-translated) for testing whether findings generalize beyond the
  MedQA/MedMCQA/MMLU-sourced ReMedQA family. Caveat: CareQA's own internal
  38 duplicate-stem groups (90 rows, different exam content per stem — see
  §6) should be handled by the pairing/cohort-construction step (owned by
  the parallel teammate working in `data/processed`), e.g. by keying on
  `unique_id` and not deduplicating by question text alone.
- **Pairing must be id-based, never positional**, both within ReMedQA's
  7-perturbation family (verified in §5) and when joining any dataset
  without a native id column (local MedQA) — content-based (normalized
  question text) joins, as used in this report, are the correct fallback
  when no id exists.

---

*Generated as part of Phase 0 dataset forensics. Source files: this
report's §7 numbers were produced live by `scripts/audit_datasets.py`;
§1–§6 numbers are cross-referenced against, and consistent with, the
independently-computed `_remedqa_findings.md` / `_careqa_findings.md` /
`_medqa_findings.md` produced by the three per-dataset audit scripts. No
value in this report was hand-fabricated.*
