"""
Forensic audit script for CareQA (English, closed-ended) — Phase 0 dataset audit.

Loads the local on-disk copy of CareQA_en (HPAI-BSC/CareQA, config CareQA_en)
from `data/raw/careqa_en` and reports schema, size, answer/options format,
category distribution, missing values, and exact-duplicate questions.

No API calls, no model training, no writes to data/processed, no modification
of the raw dataset. Output: results/audit/_careqa_findings.md
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

PROJECT_ROOT = Path(r"F:\NAACL 2027")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "careqa_en"
OUT_PATH = PROJECT_ROOT / "results" / "audit" / "_careqa_findings.md"

OPTION_COLUMN_CANDIDATES = ["op1", "op2", "op3", "op4", "op5", "op6"]
ANSWER_COLUMN_CANDIDATES = ["cop", "answer", "answer_idx", "label", "cop_idx"]
CATEGORY_COLUMN_CANDIDATES = ["category", "specialty", "topic", "subject"]
ID_COLUMN_CANDIDATES = ["id", "ID", "Id"]


def normalize_question(q):
    if q is None:
        return ""
    q = q.lower().strip()
    q = q.translate(str.maketrans("", "", string.punctuation))
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("# CareQA (English) Forensic Audit — Findings\n")
    log("Phase 0 audit only. No API calls, no training. All numbers below come")
    log("directly from the on-disk dataset at `data/raw/careqa_en`.\n")

    # -------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------
    assert DATA_PATH.exists(), f"Dataset path does not exist: {DATA_PATH}"
    ds_dict = load_from_disk(str(DATA_PATH))
    log(f"## Load\n")
    log(f"- Loaded from: `{DATA_PATH}`")
    log(f"- Splits present: {list(ds_dict.keys())}")

    if "test" not in ds_dict:
        raise RuntimeError(
            f"UNCERTAIN: expected split 'test' not found. Splits present: {list(ds_dict.keys())}"
        )

    ds = ds_dict["test"]
    n_examples = len(ds)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------
    log("\n## Schema\n")
    log(f"- n_examples (test split): **{n_examples}**")
    log("- Column names and dtypes (from `datasets.Features`):\n")
    log("| column | dtype/type |")
    log("|---|---|")
    for col_name, feature in ds.features.items():
        log(f"| `{col_name}` | `{feature}` |")

    log("\nFull `Features` repr (raw, unmodified):\n")
    log("```")
    log(repr(ds.features))
    log("```")

    columns = list(ds.features.keys())

    # -------------------------------------------------------------------
    # 5 representative full example records
    # -------------------------------------------------------------------
    log("\n## 5 Representative Example Records (raw, unmodified)\n")
    n_show = min(5, n_examples)
    # Spread picks across the dataset rather than just the first 5, to be
    # more representative; still fully raw/unmodified per-record.
    if n_examples >= n_show:
        idxs = sorted(set(int(i * (n_examples - 1) / max(n_show - 1, 1)) for i in range(n_show)))
    else:
        idxs = list(range(n_examples))
    for i in idxs:
        rec = ds[i]
        log(f"### Example (row index {i})\n")
        log("```json")
        log(json.dumps(rec, indent=2, ensure_ascii=False))
        log("```")

    # -------------------------------------------------------------------
    # Answer / label format
    # -------------------------------------------------------------------
    log("\n## Answer / Label Format\n")
    answer_col = next((c for c in ANSWER_COLUMN_CANDIDATES if c in columns), None)
    if answer_col is None:
        log(
            "UNCERTAIN: no column matching known answer-column names "
            f"({ANSWER_COLUMN_CANDIDATES}) found. Actual columns: {columns}"
        )
    else:
        vals = ds[answer_col]
        val_counter = Counter(vals)
        log(f"- Answer column: `{answer_col}`")
        log(f"- Observed dtype: `{ds.features[answer_col]}`")
        log(f"- Distinct values observed: {sorted(val_counter.keys())}")
        log(f"- Value counts: {dict(sorted(val_counter.items()))}")
        log(
            "- Interpretation: values are integers, which — combined with the "
            "`op1`..`op4` columns present in this dataset — indicates this is a "
            "1-indexed pointer into the option columns (i.e. `cop=1` -> `op1` is "
            "correct). This is an inference from the observed data layout, not "
            "an assumption from the HF dataset card; verify against the card if "
            "precision matters downstream."
        )
        # sample a few raw (question, options, cop) triples to make the
        # pointer relationship concrete/verifiable
        log("\nSample raw (question truncated, options, cop) triples for verification:\n")
        opt_cols = [c for c in OPTION_COLUMN_CANDIDATES if c in columns]
        for i in idxs[:3]:
            rec = ds[i]
            q_trunc = (rec.get("question", "") or "")[:80]
            opts = {c: rec.get(c) for c in opt_cols}
            log(f"- row {i}: question=\"{q_trunc}...\", options={opts}, {answer_col}={rec.get(answer_col)}")

    # -------------------------------------------------------------------
    # Options format
    # -------------------------------------------------------------------
    log("\n## Options Format\n")
    opt_cols = [c for c in OPTION_COLUMN_CANDIDATES if c in columns]
    if not opt_cols:
        log(
            "UNCERTAIN: no columns matching known option-column naming "
            f"conventions ({OPTION_COLUMN_CANDIDATES}) found. Actual columns: {columns}"
        )
    else:
        log(f"- Option columns found: {opt_cols}")
        log(f"- Storage: separate flat string columns (one column per option), NOT a list/array column.")
        log(f"- Count of option columns present in schema: {len(opt_cols)} (fixed by schema: every row has these same {len(opt_cols)} columns)")
        # check for empty-string options per row to see if it's effectively variable
        empty_counts_per_col = {}
        for c in opt_cols:
            vals = ds[c]
            n_empty = sum(1 for v in vals if is_missing(v))
            empty_counts_per_col[c] = n_empty
        log(f"- Empty/null values per option column: {empty_counts_per_col}")
        n_rows_with_empty_opt = sum(
            1 for i in range(n_examples) if any(is_missing(ds[c][i]) for c in opt_cols)
        ) if n_examples > 0 else 0
        # more efficient: compute via columns
        empty_flags = [
            [is_missing(v) for v in ds[c]] for c in opt_cols
        ]
        rows_with_any_empty = sum(
            1 for row_flags in zip(*empty_flags) if any(row_flags)
        )
        if all(v == 0 for v in empty_counts_per_col.values()):
            log(f"- All {n_examples} rows have all {len(opt_cols)} options populated (non-empty) -> effectively a FIXED count of {len(opt_cols)} options per question.")
        else:
            log(f"- {rows_with_any_empty} rows have at least one empty option among {opt_cols} -> option count may be effectively variable per row despite fixed schema columns.")

    # -------------------------------------------------------------------
    # Category / specialty metadata
    # -------------------------------------------------------------------
    log("\n## Category / Specialty Metadata\n")
    cat_col = next((c for c in CATEGORY_COLUMN_CANDIDATES if c in columns), None)
    if cat_col is None:
        log(
            "UNCERTAIN: no column matching known category/topic naming "
            f"conventions ({CATEGORY_COLUMN_CANDIDATES}) found. Actual columns: {columns}"
        )
    else:
        cats = ds[cat_col]
        cat_counter = Counter(cats)
        log(f"- Category column found: `{cat_col}`")
        log(f"- Number of distinct categories: {len(cat_counter)}")
        log("\n| category | count |")
        log("|---|---|")
        for cat, cnt in cat_counter.most_common():
            log(f"| {cat} | {cnt} |")

    # -------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------
    log("\n## Missing Values\n")
    check_cols = []
    if "question" in columns:
        check_cols.append("question")
    check_cols.extend(opt_cols)
    if answer_col:
        check_cols.append(answer_col)

    log("| column | n_null_or_empty |")
    log("|---|---|")
    missing_report = {}
    for c in check_cols:
        vals = ds[c]
        n_missing = sum(1 for v in vals if is_missing(v))
        missing_report[c] = n_missing
        log(f"| `{c}` | {n_missing} |")

    # answer column special case: also check for out-of-range / non-positive values
    if answer_col:
        vals = ds[answer_col]
        n_opts = len(opt_cols) if opt_cols else None
        if n_opts:
            n_out_of_range = sum(
                1 for v in vals if v is None or not (1 <= v <= n_opts)
            )
            log(f"\n- `{answer_col}` values outside expected range [1, {n_opts}]: {n_out_of_range}")

    total_missing_any = sum(missing_report.values())
    log(f"\n- Total missing/empty cells across checked columns: {total_missing_any}")

    # -------------------------------------------------------------------
    # Exact-duplicate questions (normalized)
    # -------------------------------------------------------------------
    log("\n## Exact-Duplicate Questions (normalized: lowercase, strip whitespace/punctuation)\n")
    if "question" not in columns:
        log("UNCERTAIN: no `question` column found; cannot check duplicates.")
    else:
        questions_raw = ds["question"]
        norm_to_indices = {}
        for i, q in enumerate(questions_raw):
            norm = normalize_question(q)
            norm_to_indices.setdefault(norm, []).append(i)

        dup_groups = {norm: idxs_ for norm, idxs_ in norm_to_indices.items() if len(idxs_) > 1}
        n_dup_groups = len(dup_groups)
        n_dup_rows = sum(len(v) for v in dup_groups.values())
        n_extra_rows_due_to_dup = sum(len(v) - 1 for v in dup_groups.values())

        log(f"- Distinct normalized questions: {len(norm_to_indices)}")
        log(f"- Total rows: {n_examples}")
        log(f"- Duplicate groups (normalized question appears >1 time): {n_dup_groups}")
        log(f"- Total rows involved in duplicate groups: {n_dup_rows}")
        log(f"- \"Extra\" rows beyond one-per-group (i.e. n_rows - n_distinct_normalized): {n_extra_rows_due_to_dup}")

        if dup_groups:
            log("\n### Example duplicate pairs (up to 2 groups shown)\n")
            shown = 0
            for norm, idxs_ in dup_groups.items():
                if shown >= 2:
                    break
                log(f"**Duplicate group (normalized: \"{norm[:100]}...\")** — row indices {idxs_}\n")
                for i in idxs_[:2]:
                    rec = ds[i]
                    log(f"- row {i} raw record:")
                    log("```json")
                    log(json.dumps(rec, indent=2, ensure_ascii=False))
                    log("```")
                shown += 1
        else:
            log("\nNo exact-duplicate (normalized) questions found.")

    # -------------------------------------------------------------------
    # Explicit id column check
    # -------------------------------------------------------------------
    log("\n## Explicit `id` Column Check\n")
    id_col = next((c for c in ID_COLUMN_CANDIDATES if c in columns), None)
    if id_col is None:
        log(
            f"UNCERTAIN / NOT PRESENT: no column literally named one of {ID_COLUMN_CANDIDATES} "
            f"exists in the schema. Actual columns: {columns}. "
            "Note: the dataset DOES contain `unique_id` and `exam_id` columns, which may "
            "serve an identifier-like purpose, but neither is literally named `id`. "
            "Reporting separately below rather than treating them as confirmed row identifiers."
        )
        for maybe_id_col in ["unique_id", "exam_id"]:
            if maybe_id_col in columns:
                vals = ds[maybe_id_col]
                n_unique = len(set(vals))
                log(
                    f"- `{maybe_id_col}` present: dtype=`{ds.features[maybe_id_col]}`, "
                    f"n_unique={n_unique} out of {n_examples} rows "
                    f"({'unique per row' if n_unique == n_examples else 'NOT unique per row — duplicates exist'})."
                )
    else:
        vals = ds[id_col]
        n_unique = len(set(vals))
        log(f"- `{id_col}` column present. n_unique={n_unique} out of {n_examples} rows.")

    # -------------------------------------------------------------------
    # Phrasing / style spot-check note (qualitative, for overlap-analysis teammate)
    # -------------------------------------------------------------------
    log("\n## Qualitative Note on Phrasing/Style (for overlap-analysis teammate)\n")
    log(
        "This script does not perform cross-dataset overlap analysis (that is a separate "
        "teammate's task and out of scope here). However, from the 5 representative examples "
        "printed above, reviewers should visually check whether question stems resemble "
        "USMLE/MedQA-style vignette phrasing (e.g. long clinical-vignette stems with patient "
        "age/sex/presenting-symptoms framing) versus CareQA's documented origin as Spanish "
        "MIR (Médico Interno Residente) exam questions translated to English. Any structural "
        "similarity should be flagged qualitatively by a human reviewer reading the printed "
        "examples in this file — no automated similarity score was computed here."
    )

    # -------------------------------------------------------------------
    # Write findings file
    # -------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n\nFindings written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
