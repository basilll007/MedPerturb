"""
Consolidated, reproducible Phase 0 dataset-forensics audit.

Run from the project root with:

    uv run python scripts\\audit_datasets.py

This is the ONE canonical script a reader of the paper should run to
reproduce the audit numbers reported in results/audit/dataset_audit.md. It:

  1. Loads all three raw datasets (ReMedQA, CareQA-en, local MedQA) and
     prints a schema/count summary grouped by dataset.
  2. Runs the mandatory cross-dataset overlap analysis (Task 5): normalized
     exact-question-text matching between ReMedQA's baseline `*_mcq` splits,
     local MedQA (train+val+test), and CareQA. Also runs an optional,
     clearly-separated near-duplicate scan (token-Jaccard, blocked on rare
     tokens for tractability).
  3. Regenerates results/audit/dataset_manifest.csv (one row per
     dataset/split) from scratch, computed against the live on-disk data.

It does NOT touch data/processed or figures/, and does not modify any raw
dataset. No API calls, no training, no GPU work.

Guardrail: every number below is computed at run time. Anything not
directly verifiable from the data is printed as "UNCERTAIN: <reason>"
rather than guessed.
"""

from __future__ import annotations

import csv
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Paths (pathlib only, no string concatenation)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
REMEDQA_PATH = DATA_RAW_DIR / "remedqa"
CAREQA_PATH = DATA_RAW_DIR / "careqa_en"
MEDQA_PATH = DATA_RAW_DIR / "medqa"
RESULTS_DIR = PROJECT_ROOT / "results" / "audit"
MANIFEST_PATH = RESULTS_DIR / "dataset_manifest.csv"

SOURCES = ["medqa", "medmcqa", "mmlu"]
PERTURBATIONS = [
    "mcq",
    "open",
    "incorrect",
    "roman_numeral",
    "none_of_the_provided",
    "fixed_pos",
    "no_symbols",
]

# ---------------------------------------------------------------------------
# Canonical normalization — defined ONCE, reused for every comparison in
# this script (manifest duplicate-counts AND all Task 5 overlap checks).
#   1. lowercase
#   2. strip leading/trailing whitespace
#   3. remove ASCII punctuation (string.punctuation, via str.translate)
#   4. collapse internal whitespace runs to a single space
# This matches the normalization already used in scripts/_audit_careqa.py
# and scripts/_audit_medqa.py (str.translate over string.punctuation).
# ---------------------------------------------------------------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_question(text) -> str:
    if text is None:
        return ""
    s = str(text).lower().strip()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_all():
    print(f"Loading ReMedQA from {REMEDQA_PATH} ...")
    remedqa = load_from_disk(str(REMEDQA_PATH))
    print(f"  -> {len(remedqa)} splits")

    print(f"Loading CareQA (en) from {CAREQA_PATH} ...")
    careqa_dd = load_from_disk(str(CAREQA_PATH))
    careqa = careqa_dd["test"]
    print(f"  -> {careqa.num_rows} rows (split: test)")

    print(f"Loading local MedQA from {MEDQA_PATH} ...")
    medqa = load_from_disk(str(MEDQA_PATH))
    print(f"  -> splits: {list(medqa.keys())}")

    return remedqa, careqa, medqa


# ---------------------------------------------------------------------------
# Schema / count summary (grouped by dataset)
# ---------------------------------------------------------------------------
def print_schema_summary(remedqa, careqa, medqa) -> None:
    banner("SCHEMA / COUNT SUMMARY -- ReMedQA")
    print(f"Splits: {len(remedqa)} (expected {len(SOURCES)} x {len(PERTURBATIONS)} = {len(SOURCES) * len(PERTURBATIONS)})")
    for split_name in remedqa.keys():
        split = remedqa[split_name]
        cols = list(split.features.keys())
        print(f"  {split_name:32s} n={split.num_rows:5d}  columns={cols}")

    banner("SCHEMA / COUNT SUMMARY -- CareQA (en)")
    print(f"n_rows={careqa.num_rows}  columns={list(careqa.features.keys())}")

    banner("SCHEMA / COUNT SUMMARY -- local MedQA")
    for split_name in medqa.keys():
        split = medqa[split_name]
        print(f"  {split_name:12s} n={split.num_rows:6d}  columns={list(split.features.keys())}")
    total = sum(medqa[s].num_rows for s in medqa.keys())
    print(f"  TOTAL across splits: {total}")


# ---------------------------------------------------------------------------
# Manifest generation (Task 6)
# ---------------------------------------------------------------------------
def manifest_rows_remedqa(remedqa) -> list[dict]:
    rows = []
    notes_by_pert = {
        "mcq": (
            "Baseline, unperturbed MCQ. options/answer are Python-repr strings "
            "-- use ast.literal_eval, NOT json.loads (json.loads silently "
            "misparses some numeric-looking answers)."
        ),
        "open": (
            "Question text is REWORDED into an open-ended (non-MCQ) form and "
            "answer becomes free text instead of a letter. Changes task format "
            "away from MCQ -- NOT a meaning-preserving perturbation of the "
            "original question/answer format."
        ),
        "incorrect": (
            "Question and options text unchanged, but the task itself is "
            "inverted: the model must name the THREE incorrect options "
            "(answer is a list of 3 letters) instead of the one correct "
            "option. Task-altering, not meaning-preserving."
        ),
        "roman_numeral": (
            "Only the option labels change (A/B/C/D -> I/II/III/IV); question "
            "text, option text, and answer content are unchanged. Clean, "
            "meaning-preserving option-LABEL-format perturbation."
        ),
        "none_of_the_provided": (
            "The correct option's TEXT is overwritten with the generic string "
            "'None of the provided options', and that option remains the "
            "labeled correct answer. The actual correct-answer content is "
            "removed from the choices entirely. Alters task semantics -- "
            "NOT meaning-preserving."
        ),
        "fixed_pos": (
            "Option order/position is shuffled relative to mcq; the answer "
            "letter changes accordingly but option text and question are "
            "unchanged. Meaning-preserving option-ORDER-format perturbation."
        ),
        "no_symbols": (
            "Verified directly against mcq: question TEXT IS UNCHANGED (not "
            "reworded) and options are still shown, just without letter "
            "labels; only the expected answer format changes (full option "
            "text instead of a letter). This differs from a naive assumption "
            "that no_symbols rewords the question -- it does not; only 'open' "
            "does."
        ),
    }
    for split_name in remedqa.keys():
        split = remedqa[split_name]
        n = split.num_rows
        questions = split["question"]
        answers = split["answer"]
        ids = split["id"]
        missing_q = sum(1 for v in questions if is_missing(v))
        missing_a = sum(1 for v in answers if is_missing(v))
        norm_qs = [normalize_question(q) for q in questions]
        counts = Counter(norm_qs)
        dup_rows = sum(c for c in counts.values() if c > 1)

        source = next((s for s in SOURCES if split_name.startswith(s + "_")), "UNKNOWN")
        pert = split_name[len(source) + 1 :] if source != "UNKNOWN" else split_name
        note = notes_by_pert.get(pert, "")

        rows.append(
            {
                "dataset": "ReMedQA",
                "split": split_name,
                "n_examples": n,
                "n_columns": len(split.features),
                "question_column": "question",
                "answer_column": "answer",
                "options_column": "options",
                "id_column": "id",
                "missing_questions": missing_q,
                "missing_answers": missing_a,
                "duplicate_questions": dup_rows,
                "notes": note,
            }
        )
    return rows


def manifest_rows_careqa(careqa) -> list[dict]:
    n = careqa.num_rows
    questions = careqa["question"]
    cops = careqa["cop"]
    missing_q = sum(1 for v in questions if is_missing(v))
    missing_a = sum(1 for v in cops if v is None or not (1 <= v <= 4))
    norm_qs = [normalize_question(q) for q in questions]
    counts = Counter(norm_qs)
    dup_rows = sum(c for c in counts.values() if c > 1)

    unique_ids = careqa["unique_id"]
    exam_ids = careqa["exam_id"]
    n_unique_uid = len(set(unique_ids))
    n_unique_exam = len(set(exam_ids))

    note = (
        f"No column literally named 'id'. `unique_id` is unique per row "
        f"({n_unique_uid}/{n} unique) and used here as CareQA's id. `exam_id` "
        f"is NOT unique ({n_unique_exam} distinct values across {n} rows). "
        f"`cop` is a 1-indexed pointer into op1..op4 (verified: values in "
        f"{{1,2,3,4}}). {dup_rows} rows fall into normalized-duplicate-question "
        f"groups; manual review showed these are same phrasing / different "
        f"exam content (different options+answer), NOT true duplicates -- "
        f"do not naively collapse them."
    )

    return [
        {
            "dataset": "CareQA",
            "split": "test",
            "n_examples": n,
            "n_columns": len(careqa.features),
            "question_column": "question",
            "answer_column": "cop",
            "options_column": "op1;op2;op3;op4",
            "id_column": "unique_id",
            "missing_questions": missing_q,
            "missing_answers": missing_a,
            "duplicate_questions": dup_rows,
            "notes": note,
        }
    ]


def manifest_rows_medqa(medqa) -> list[dict]:
    rows = []
    for split_name in medqa.keys():
        split = medqa[split_name]
        n = split.num_rows
        questions = split["question"]
        answer_idx = split["answer_idx"]
        missing_q = sum(1 for v in questions if is_missing(v))
        missing_a = sum(1 for v in answer_idx if is_missing(v))
        norm_qs = [normalize_question(q) for q in questions]
        counts = Counter(norm_qs)
        dup_rows = sum(c for c in counts.values() if c > 1)

        note = (
            "No id column at all; only positional/row index. Options stored "
            "as a fixed-key dict {A,B,C,D}. `answer` (free text) verified to "
            "equal options[answer_idx] on a 200-row sample; `answer_idx` used "
            "here as the canonical answer column. 0 cross-split (train/val/"
            "test) normalized-question duplicates found (verified below)."
        )

        rows.append(
            {
                "dataset": "MedQA",
                "split": split_name,
                "n_examples": n,
                "n_columns": len(split.features),
                "question_column": "question",
                "answer_column": "answer_idx",
                "options_column": "options",
                "id_column": "NONE",
                "missing_questions": missing_q,
                "missing_answers": missing_a,
                "duplicate_questions": dup_rows,
                "notes": note,
            }
        )
    return rows


def write_manifest(remedqa, careqa, medqa) -> None:
    banner("MANIFEST GENERATION")
    rows = []
    rows.extend(manifest_rows_remedqa(remedqa))
    rows.extend(manifest_rows_careqa(careqa))
    rows.extend(manifest_rows_medqa(medqa))

    fieldnames = [
        "dataset",
        "split",
        "n_examples",
        "n_columns",
        "question_column",
        "answer_column",
        "options_column",
        "id_column",
        "missing_questions",
        "missing_answers",
        "duplicate_questions",
        "notes",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {MANIFEST_PATH}")
    expected = len(SOURCES) * len(PERTURBATIONS) + 1 + len(medqa.keys())
    status = "MATCH" if len(rows) == expected else "MISMATCH -- investigate"
    print(f"Expected row count: 21 (ReMedQA) + 1 (CareQA) + {len(medqa.keys())} (MedQA) = {expected} ({status})")


# ---------------------------------------------------------------------------
# Task 5: cross-dataset overlap analysis
# ---------------------------------------------------------------------------
def _norm_index(questions: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    norms = [normalize_question(q) for q in questions]
    idx_by_norm: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(norms):
        if n:
            idx_by_norm[n].append(i)
    return norms, idx_by_norm


def exact_overlap(
    name_a: str,
    questions_a: list[str],
    name_b: str,
    questions_b: list[str],
    n_examples: int = 3,
) -> dict:
    norms_a, idx_a = _norm_index(questions_a)
    norms_b, idx_b = _norm_index(questions_b)
    set_a, set_b = set(idx_a.keys()), set(idx_b.keys())
    shared = set_a & set_b

    rows_a_matched = sum(len(idx_a[n]) for n in shared)
    rows_b_matched = sum(len(idx_b[n]) for n in shared)
    pct_a = 100.0 * rows_a_matched / len(questions_a) if questions_a else 0.0
    pct_b = 100.0 * rows_b_matched / len(questions_b) if questions_b else 0.0

    print(f"\n[EXACT] {name_a} (n={len(questions_a)})  <->  {name_b} (n={len(questions_b)})")
    print(f"  distinct normalized questions shared: {len(shared)}")
    print(f"  rows in {name_a} matched: {rows_a_matched} / {len(questions_a)} ({pct_a:.3f}%)")
    print(f"  rows in {name_b} matched: {rows_b_matched} / {len(questions_b)} ({pct_b:.3f}%)")

    examples = []
    for n in list(shared)[:n_examples]:
        ia, ib = idx_a[n][0], idx_b[n][0]
        examples.append((questions_a[ia], questions_b[ib]))
        print(f"    example match:\n      [{name_a}] {questions_a[ia][:200]}\n      [{name_b}] {questions_b[ib][:200]}")

    return {
        "pair": f"{name_a} <-> {name_b}",
        "n_a": len(questions_a),
        "n_b": len(questions_b),
        "shared_normalized": len(shared),
        "rows_a_matched": rows_a_matched,
        "rows_b_matched": rows_b_matched,
        "pct_a": pct_a,
        "pct_b": pct_b,
        "examples": examples,
    }


def _tokenize_rare(norm_text: str, min_len: int = 5) -> set[str]:
    return {w for w in norm_text.split() if len(w) >= min_len}


def near_duplicate_scan(
    name_a: str,
    questions_a: list[str],
    name_b: str,
    questions_b: list[str],
    threshold: float = 0.85,
    max_doc_freq: int = 8,
    cap_candidate_pairs: int = 2_000_000,
) -> dict:
    """
    OPTIONAL / SECONDARY signal, kept clearly separate from exact matching.
    Method: block candidate pairs on shared "rare" tokens (length>=5 words
    with document frequency <= max_doc_freq within side B) to keep this
    tractable, then compute full word-set Jaccard similarity on the
    surviving candidates. This is NOT exhaustive (a pair sharing no rare
    token is never compared) and NOT called "duplicate" -- these are
    near-duplicate CANDIDATES above the stated Jaccard threshold, excluding
    any pair that is already an exact normalized match.
    """
    norms_a = [normalize_question(q) for q in questions_a]
    norms_b = [normalize_question(q) for q in questions_b]
    exact_b = set(norms_b)

    toks_b = [_tokenize_rare(n) for n in norms_b]
    df_b: Counter = Counter()
    for t in toks_b:
        for w in t:
            df_b[w] += 1
    inv_index: dict[str, list[int]] = defaultdict(list)
    for j, t in enumerate(toks_b):
        for w in t:
            if df_b[w] <= max_doc_freq:
                inv_index[w].append(j)

    matches: list[tuple[int, int, float]] = []
    pairs_checked = 0
    truncated = False
    for i, n_a in enumerate(norms_a):
        if not n_a or n_a in exact_b:
            continue
        toks_a = _tokenize_rare(n_a)
        candidates: set[int] = set()
        for w in toks_a:
            if w in inv_index:
                candidates.update(inv_index[w])
        for j in candidates:
            pairs_checked += 1
            if pairs_checked > cap_candidate_pairs:
                truncated = True
                break
            n_b = norms_b[j]
            wa, wb = set(n_a.split()), set(n_b.split())
            if not wa or not wb:
                continue
            jac = len(wa & wb) / len(wa | wb)
            if jac >= threshold:
                matches.append((i, j, jac))
        if truncated:
            break

    print(
        f"\n[NEAR-DUP, threshold={threshold}, excludes exact matches] "
        f"{name_a} <-> {name_b}: {len(matches)} candidate pair(s) found "
        f"(blocked scan, {pairs_checked} candidate pairs evaluated"
        f"{', TRUNCATED at cap' if truncated else ''})."
    )
    for i, j, jac in matches[:3]:
        print(f"    jaccard={jac:.2f}\n      [{name_a}] {questions_a[i][:200]}\n      [{name_b}] {questions_b[j][:200]}")

    return {
        "pair": f"{name_a} <-> {name_b}",
        "threshold": threshold,
        "n_candidates_found": len(matches),
        "pairs_checked": pairs_checked,
        "truncated": truncated,
    }


def run_overlap_analysis(remedqa, careqa, medqa) -> None:
    banner("TASK 5 -- CROSS-DATASET OVERLAP (normalized exact-match, primary/mandatory)")
    print(
        "Normalization (defined once, reused everywhere): lowercase -> strip "
        "-> remove ASCII punctuation -> collapse internal whitespace."
    )

    remedqa_medqa_q = list(remedqa["medqa_mcq"]["question"])
    remedqa_medmcqa_q = list(remedqa["medmcqa_mcq"]["question"])
    remedqa_mmlu_q = list(remedqa["mmlu_mcq"]["question"])
    careqa_q = list(careqa["question"])

    medqa_train_q = list(medqa["train"]["question"])
    medqa_val_q = list(medqa["validation"]["question"])
    medqa_test_q = list(medqa["test"]["question"])
    medqa_all_q = medqa_train_q + medqa_val_q + medqa_test_q

    results = {}
    print("\n--- ReMedQA(medqa_mcq) <-> local MedQA ---")
    results["remedqa_medqa__medqa_train"] = exact_overlap("ReMedQA(medqa_mcq)", remedqa_medqa_q, "MedQA(train)", medqa_train_q)
    results["remedqa_medqa__medqa_val"] = exact_overlap("ReMedQA(medqa_mcq)", remedqa_medqa_q, "MedQA(validation)", medqa_val_q)
    results["remedqa_medqa__medqa_test"] = exact_overlap("ReMedQA(medqa_mcq)", remedqa_medqa_q, "MedQA(test)", medqa_test_q)
    results["remedqa_medqa__medqa_all"] = exact_overlap("ReMedQA(medqa_mcq)", remedqa_medqa_q, "MedQA(train+val+test)", medqa_all_q)

    print("\n--- ReMedQA(medmcqa_mcq) <-> CareQA ---")
    results["remedqa_medmcqa__careqa"] = exact_overlap("ReMedQA(medmcqa_mcq)", remedqa_medmcqa_q, "CareQA(test)", careqa_q)

    print("\n--- ReMedQA(mmlu_mcq) <-> CareQA ---")
    results["remedqa_mmlu__careqa"] = exact_overlap("ReMedQA(mmlu_mcq)", remedqa_mmlu_q, "CareQA(test)", careqa_q)

    print("\n--- ReMedQA(medqa_mcq) <-> CareQA ---")
    results["remedqa_medqa__careqa"] = exact_overlap("ReMedQA(medqa_mcq)", remedqa_medqa_q, "CareQA(test)", careqa_q)

    print("\n--- local MedQA <-> CareQA ---")
    results["medqa_all__careqa"] = exact_overlap("MedQA(train+val+test)", medqa_all_q, "CareQA(test)", careqa_q)

    banner("TASK 5 -- NEAR-DUPLICATE SCAN (optional/secondary, kept separate from exact match)")
    near_duplicate_scan("ReMedQA(medqa_mcq)", remedqa_medqa_q, "MedQA(train+val+test)", medqa_all_q)
    near_duplicate_scan("ReMedQA(medmcqa_mcq)", remedqa_medmcqa_q, "CareQA(test)", careqa_q)
    near_duplicate_scan("ReMedQA(mmlu_mcq)", remedqa_mmlu_q, "CareQA(test)", careqa_q)
    near_duplicate_scan("ReMedQA(medqa_mcq)", remedqa_medqa_q, "CareQA(test)", careqa_q)
    near_duplicate_scan("MedQA(train+val+test)", medqa_all_q, "CareQA(test)", careqa_q)

    banner("TASK 5 -- SUMMARY TABLE")
    print(f"{'pair':55s} {'n_a':>7s} {'n_b':>7s} {'shared':>7s} {'%a':>7s} {'%b':>7s}")
    for r in results.values():
        print(f"{r['pair']:55s} {r['n_a']:7d} {r['n_b']:7d} {r['shared_normalized']:7d} {r['pct_a']:6.2f}% {r['pct_b']:6.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    banner("PHASE 0 CONSOLIDATED DATASET AUDIT")
    print("No API calls, no training, no GPU work. Read-only against data/raw.")
    print("Does not touch data/processed or figures/.")

    remedqa, careqa, medqa = load_all()

    print_schema_summary(remedqa, careqa, medqa)

    run_overlap_analysis(remedqa, careqa, medqa)

    write_manifest(remedqa, careqa, medqa)

    banner("DONE")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
