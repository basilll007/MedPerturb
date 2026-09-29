"""
Forensic audit of the ReMedQA dataset (Phase 0 — dataset audit only).

Loads the local `datasets.DatasetDict` at data/raw/remedqa and produces a
grounded markdown report at results/audit/_remedqa_findings.md covering:

  1. Split structure, columns/dtypes, representative raw + parsed examples.
  2. Duplicate `id` checks within each split.
  3. Cross-perturbation id-alignment per source family (medqa/medmcqa/mmlu):
     is the same set of ids present across all 7 perturbations of a source?
  4. Side-by-side inspection of ~5 shared ids per source across all 7
     perturbations, to visually verify meaning-preserving pairing and see
     how `answer` changes format/value across perturbations.
  5. Index-based vs id-based join comparison (does row order happen to
     align, i.e. would a naive positional join accidentally work?).
  6. Missing/null question/options/answer checks.
  7. Exact-duplicate (normalized) question checks within each split.
  8. A descriptive summary of what each of the 7 perturbation types
     actually does, based on direct before/after inspection against the
     `mcq` split for the same source.

This script does NOT modify data/raw or write to data/processed. It is
read-only against the raw dataset and only writes the findings markdown
file plus stdout logging.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

# ---------------------------------------------------------------------------
# Fixed paths (no hardcoded string concatenation)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "remedqa"
RESULTS_DIR = PROJECT_ROOT / "results" / "audit"
FINDINGS_PATH = RESULTS_DIR / "_remedqa_findings.md"

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
N_SHARED_ID_SAMPLES = 5
N_REPRESENTATIVE_EXAMPLES = 2

# ---------------------------------------------------------------------------
# Report buffer — every finding is appended here with real computed values.
# ---------------------------------------------------------------------------
REPORT_LINES: list[str] = []


def emit(line: str = "") -> None:
    """Append a line to the markdown report buffer (and echo to stdout)."""
    REPORT_LINES.append(line)
    print(line)


def try_json_parse(value: Any) -> tuple[Any, str]:
    """Attempt json.loads on a value; report the actual observed format."""
    if value is None:
        return None, "null"
    if not isinstance(value, str):
        return value, f"non-string ({type(value).__name__})"
    try:
        parsed = json.loads(value)
        return parsed, f"json ({type(parsed).__name__})"
    except (json.JSONDecodeError, TypeError):
        return value, "raw-string (not valid JSON)"


def normalize_question(text: Any) -> str:
    """Lowercase, strip whitespace/punctuation for near-duplicate detection."""
    if not isinstance(text, str):
        return ""
    out = text.lower().strip()
    out = "".join(ch for ch in out if ch.isalnum() or ch.isspace())
    out = " ".join(out.split())
    return out


def load_dataset() -> DatasetDict:
    ds = load_from_disk(str(RAW_DATA_DIR))
    if not isinstance(ds, DatasetDict):
        raise TypeError(
            f"Expected a DatasetDict at {RAW_DATA_DIR}, got {type(ds).__name__}"
        )
    return ds


def source_of_split(split_name: str) -> str | None:
    for source in SOURCES:
        if split_name.startswith(source + "_"):
            return source
    return None


def perturbation_of_split(split_name: str, source: str) -> str:
    return split_name[len(source) + 1 :]


def section_split_overview(ds: DatasetDict) -> None:
    emit("## 1. Split Structure, Columns, and Representative Examples")
    emit()
    emit(f"Total splits: **{len(ds)}**")
    emit()
    emit("| split | n_examples | columns |")
    emit("|---|---|---|")
    for split_name in ds.keys():
        split = ds[split_name]
        cols = ", ".join(f"{c}:{t}" for c, t in split.features.items())
        emit(f"| {split_name} | {split.num_rows} | {cols} |")
    emit()

    for split_name in ds.keys():
        split = ds[split_name]
        emit(f"### Split: `{split_name}`")
        emit()
        emit(f"- n_examples: {split.num_rows}")
        emit(f"- columns/dtypes: {dict((c, str(t)) for c, t in split.features.items())}")
        emit()
        n_show = min(N_REPRESENTATIVE_EXAMPLES, split.num_rows)
        for i in range(n_show):
            row = split[i]
            emit(f"**Raw example [{i}] (unparsed):**")
            emit("```json")
            emit(json.dumps(row, indent=2, ensure_ascii=False))
            emit("```")

            parsed_options, options_fmt = try_json_parse(row.get("options"))
            parsed_answer, answer_fmt = try_json_parse(row.get("answer"))
            emit(f"**Parsed example [{i}]:**")
            emit(f"- `options` observed format: {options_fmt}")
            emit(f"- `options` parsed: `{parsed_options}`")
            emit(f"- `answer` observed format: {answer_fmt}")
            emit(f"- `answer` parsed: `{parsed_answer}`")
            emit()


def section_duplicate_ids(ds: DatasetDict) -> dict[str, list[str]]:
    emit("## 2. Duplicate `id` Checks Within Each Split")
    emit()
    dup_report: dict[str, list[str]] = {}
    emit("| split | n_rows | n_unique_ids | n_duplicate_id_values |")
    emit("|---|---|---|---|")
    for split_name in ds.keys():
        ids = ds[split_name]["id"]
        counts = Counter(ids)
        dupes = [i for i, c in counts.items() if c > 1]
        dup_report[split_name] = dupes
        emit(f"| {split_name} | {len(ids)} | {len(counts)} | {len(dupes)} |")
    emit()
    any_dupes = any(dupes for dupes in dup_report.values())
    if any_dupes:
        emit("**Splits with duplicate ids and example duplicated values:**")
        for split_name, dupes in dup_report.items():
            if dupes:
                emit(f"- `{split_name}`: {len(dupes)} duplicated id value(s), e.g. {dupes[:5]}")
    else:
        emit("No duplicate `id` values found in any split.")
    emit()
    return dup_report


def section_cross_perturbation_alignment(
    ds: DatasetDict,
) -> dict[str, dict[str, Any]]:
    emit("## 3. Cross-Perturbation ID Alignment Per Source Family")
    emit()
    alignment_info: dict[str, dict[str, Any]] = {}

    for source in SOURCES:
        emit(f"### Source: `{source}`")
        emit()
        id_sets: dict[str, set[str]] = {}
        for pert in PERTURBATIONS:
            split_name = f"{source}_{pert}"
            if split_name not in ds:
                emit(f"UNCERTAIN: split `{split_name}` not found in DatasetDict — skipping.")
                continue
            id_sets[pert] = set(ds[split_name]["id"])

        if len(id_sets) != len(PERTURBATIONS):
            emit(
                f"UNCERTAIN: expected {len(PERTURBATIONS)} perturbation splits for "
                f"source `{source}`, found {len(id_sets)}. Alignment analysis below "
                f"uses only the splits found."
            )

        all_ids_union = set().union(*id_sets.values()) if id_sets else set()
        all_ids_intersection = (
            set.intersection(*id_sets.values()) if id_sets else set()
        )

        # per-perturbation counts
        emit("| perturbation | n_ids_in_split |")
        emit("|---|---|")
        for pert, s in id_sets.items():
            emit(f"| {pert} | {len(s)} |")
        emit()

        emit(f"- Union of ids across all {len(id_sets)} perturbations: **{len(all_ids_union)}**")
        emit(
            f"- Intersection (ids common to ALL {len(id_sets)} perturbations): "
            f"**{len(all_ids_intersection)}**"
        )

        # ids missing from at least one perturbation (i.e. in union but not intersection)
        missing_from_some = all_ids_union - all_ids_intersection
        emit(f"- Ids present in union but missing from at least one perturbation: **{len(missing_from_some)}**")

        # per-id count of how many perturbations it appears in, to report "only in some"
        id_membership_count: Counter = Counter()
        for s in id_sets.values():
            id_membership_count.update(s)
        appears_in_all = sum(1 for c in id_membership_count.values() if c == len(id_sets))
        appears_in_some_not_all = sum(
            1 for c in id_membership_count.values() if 0 < c < len(id_sets)
        )
        emit(
            f"- Cross-check: ids appearing in all {len(id_sets)} perturbation splits: "
            f"**{appears_in_all}** (should equal intersection count above: "
            f"{'MATCH' if appears_in_all == len(all_ids_intersection) else 'MISMATCH — investigate'})"
        )
        emit(f"- Ids appearing in only SOME (not all) perturbation splits: **{appears_in_some_not_all}**")
        emit()

        alignment_info[source] = {
            "id_sets": id_sets,
            "intersection": all_ids_intersection,
            "union": all_ids_union,
        }

    return alignment_info


def section_shared_id_sidebyside(
    ds: DatasetDict, alignment_info: dict[str, dict[str, Any]]
) -> None:
    emit("## 4. Side-by-Side Inspection of Shared IDs Across All 7 Perturbations")
    emit()
    for source in SOURCES:
        info = alignment_info.get(source)
        if not info:
            emit(f"UNCERTAIN: no alignment info for source `{source}`.")
            continue
        shared_ids = sorted(info["intersection"])
        if not shared_ids:
            emit(
                f"### Source: `{source}` — NO ids shared across all 7 perturbations. "
                f"Skipping side-by-side inspection (id-based pairing does not hold)."
            )
            emit()
            continue

        sample_ids = shared_ids[:N_SHARED_ID_SAMPLES]
        emit(f"### Source: `{source}` — sample of {len(sample_ids)} shared ids")
        emit()

        # Build id -> row lookup per perturbation for the sampled ids only.
        for pert in PERTURBATIONS:
            split_name = f"{source}_{pert}"
            if split_name not in ds:
                continue
        for sample_id in sample_ids:
            emit(f"#### id = `{sample_id}`")
            emit()
            emit("| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |")
            emit("|---|---|---|---|---|")
            for pert in PERTURBATIONS:
                split_name = f"{source}_{pert}"
                if split_name not in ds:
                    continue
                split = ds[split_name]
                # find row with matching id (dataset not guaranteed sorted by id)
                matches = [r for r in split if r["id"] == sample_id]
                if not matches:
                    emit(f"| {pert} | UNCERTAIN: id not found (unexpected — should be in intersection) | | | |")
                    continue
                row = matches[0]
                q = (row.get("question") or "").replace("\n", " ").replace("|", "\\|")
                q_trunc = q[:200] + ("..." if len(q) > 200 else "")
                parsed_opts, _ = try_json_parse(row.get("options"))
                parsed_ans, _ = try_json_parse(row.get("answer"))
                opts_str = json.dumps(parsed_opts, ensure_ascii=False).replace("|", "\\|")
                if len(opts_str) > 200:
                    opts_str = opts_str[:200] + "..."
                raw_ans = str(row.get("answer")).replace("|", "\\|")
                parsed_ans_str = json.dumps(parsed_ans, ensure_ascii=False).replace("|", "\\|")
                emit(f"| {pert} | {q_trunc} | {opts_str} | {raw_ans} | {parsed_ans_str} |")
            emit()


def section_index_vs_id_join(ds: DatasetDict, alignment_info: dict[str, dict[str, Any]]) -> None:
    emit("## 5. Index-Based vs ID-Based Join Stability")
    emit()
    emit(
        "For one source family, compares how many rows a naive positional "
        "(index-based) join across two perturbation splits would agree with "
        "vs. how many an id-based join actually pairs correctly."
    )
    emit()

    # Use medqa as the representative source family per task instructions,
    # but run for all three for completeness/rigor.
    for source in SOURCES:
        info = alignment_info.get(source)
        if not info:
            continue
        split_a_name = f"{source}_mcq"
        split_b_name = f"{source}_open"
        if split_a_name not in ds or split_b_name not in ds:
            emit(f"UNCERTAIN: could not run index-vs-id join for `{source}` — missing split.")
            continue
        split_a = ds[split_a_name]
        split_b = ds[split_b_name]

        n_a, n_b = split_a.num_rows, split_b.num_rows
        same_length = n_a == n_b
        emit(f"### Source: `{source}` (`{split_a_name}` vs `{split_b_name}`)")
        emit(f"- n_rows: {n_a} vs {n_b} ({'equal' if same_length else 'DIFFERENT — index join impossible/misaligned by construction'})")

        if not same_length:
            emit("- Skipping positional comparison since lengths differ.")
            emit()
            continue

        ids_a = split_a["id"]
        ids_b = split_b["id"]
        index_join_matches = sum(1 for a, b in zip(ids_a, ids_b) if a == b)
        emit(
            f"- Positional (index-based) join: rows where id at position i is IDENTICAL "
            f"between `{split_a_name}` and `{split_b_name}`: **{index_join_matches} / {n_a}** "
            f"({100 * index_join_matches / n_a:.2f}%)"
        )

        id_based_join_count = len(set(ids_a) & set(ids_b))
        emit(
            f"- ID-based join: ids present in BOTH splits (regardless of position): "
            f"**{id_based_join_count} / {n_a}** ({100 * id_based_join_count / n_a:.2f}%)"
        )

        if index_join_matches == id_based_join_count == n_a:
            emit(
                "- CONCLUSION: row order is fully stable — a naive positional join "
                "would happen to produce the SAME result as an id-based join for this pair."
            )
        elif index_join_matches == 0:
            emit(
                "- CONCLUSION: row order is NOT preserved at all between these splits — "
                "a positional join would silently pair unrelated questions. ID-based join is required."
            )
        else:
            emit(
                f"- CONCLUSION: row order is PARTIALLY preserved ({index_join_matches} of "
                f"{id_based_join_count} id-matched rows are also positionally aligned). "
                "A positional join would be silently wrong for the remainder. ID-based join is required."
            )
        emit()


def section_missing_values(ds: DatasetDict) -> None:
    emit("## 6. Missing / Null Field Checks")
    emit()
    emit("| split | null_question | empty_question | null_options | empty_options | null_answer | empty_answer |")
    emit("|---|---|---|---|---|---|---|")
    for split_name in ds.keys():
        split = ds[split_name]
        q = split["question"]
        o = split["options"]
        a = split["answer"]
        null_q = sum(1 for v in q if v is None)
        empty_q = sum(1 for v in q if isinstance(v, str) and v.strip() == "")
        null_o = sum(1 for v in o if v is None)
        empty_o = sum(1 for v in o if isinstance(v, str) and v.strip() == "")
        null_a = sum(1 for v in a if v is None)
        empty_a = sum(1 for v in a if isinstance(v, str) and v.strip() == "")
        emit(f"| {split_name} | {null_q} | {empty_q} | {null_o} | {empty_o} | {null_a} | {empty_a} |")
    emit()


def section_duplicate_questions(ds: DatasetDict) -> None:
    emit("## 7. Exact-Duplicate (Normalized) Questions Within Each Split")
    emit()
    emit("Normalization: lowercase, strip punctuation, collapse whitespace.")
    emit()
    emit("| split | n_rows | n_unique_normalized_questions | n_rows_involved_in_duplicates |")
    emit("|---|---|---|---|")
    for split_name in ds.keys():
        split = ds[split_name]
        norm_qs = [normalize_question(q) for q in split["question"]]
        counts = Counter(norm_qs)
        dup_rows = sum(c for c in counts.values() if c > 1)
        emit(f"| {split_name} | {len(norm_qs)} | {len(counts)} | {dup_rows} |")
    emit()


def section_perturbation_semantics(ds: DatasetDict, alignment_info: dict[str, dict[str, Any]]) -> None:
    emit("## 8. What Each Perturbation Type Actually Does (Direct Inspection)")
    emit()
    emit(
        "Method: for each source, take one id shared across all 7 perturbations, "
        "and diff the `mcq` version's `options`/`answer` against each other "
        "perturbation's `options`/`answer` for that exact same id."
    )
    emit()

    for source in SOURCES:
        info = alignment_info.get(source)
        if not info or not info["intersection"]:
            emit(f"UNCERTAIN: source `{source}` has no fully-shared id to inspect perturbation semantics with.")
            continue
        sample_id = sorted(info["intersection"])[0]
        emit(f"### Source: `{source}`, inspected id = `{sample_id}`")
        emit()

        rows_by_pert: dict[str, dict] = {}
        for pert in PERTURBATIONS:
            split_name = f"{source}_{pert}"
            if split_name not in ds:
                continue
            split = ds[split_name]
            matches = [r for r in split if r["id"] == sample_id]
            if matches:
                rows_by_pert[pert] = matches[0]

        mcq_row = rows_by_pert.get("mcq")
        if not mcq_row:
            emit("UNCERTAIN: mcq baseline row not found for this id.")
            continue

        mcq_opts, _ = try_json_parse(mcq_row.get("options"))
        mcq_ans, _ = try_json_parse(mcq_row.get("answer"))
        emit(f"- **mcq (baseline)**: options=`{mcq_opts}`, answer=`{mcq_ans}` (raw: `{mcq_row.get('answer')}`)")

        for pert in PERTURBATIONS:
            if pert == "mcq":
                continue
            row = rows_by_pert.get(pert)
            if not row:
                emit(f"- **{pert}**: UNCERTAIN — row not found for this id.")
                continue
            opts, _ = try_json_parse(row.get("options"))
            ans, _ = try_json_parse(row.get("answer"))
            emit(f"- **{pert}**: options=`{opts}`, answer=`{ans}` (raw: `{row.get('answer')}`)")
        emit()

    emit("### Interpretation notes (grounded in the printed diffs above — verify against actual output before trusting)")
    emit()
    emit(
        "- These are observations from ONE sampled id per source; see Section 4 for "
        "additional samples. Treat perturbation-level generalizations as provisional "
        "unless consistent across all inspected samples."
    )
    emit()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    emit("# ReMedQA Dataset Forensic Audit")
    emit()
    emit(f"Source data: `{RAW_DATA_DIR}`")
    emit()
    emit(
        "This report is generated programmatically by `scripts/_audit_remedqa.py` "
        "from the actual on-disk dataset. All numbers below are computed at run time, "
        "not fabricated. Any ambiguous finding is marked `UNCERTAIN: <reason>`."
    )
    emit()

    ds = load_dataset()

    emit(f"Expected splits (source x perturbation, {len(SOURCES)}x{len(PERTURBATIONS)}): {len(SOURCES) * len(PERTURBATIONS)}")
    emit(f"Actual splits found: {len(ds)}")
    unexpected = [s for s in ds.keys() if source_of_split(s) is None]
    if unexpected:
        emit(f"UNCERTAIN: splits with unrecognized source prefix: {unexpected}")
    emit()

    section_split_overview(ds)
    section_duplicate_ids(ds)
    alignment_info = section_cross_perturbation_alignment(ds)
    section_shared_id_sidebyside(ds, alignment_info)
    section_index_vs_id_join(ds, alignment_info)
    section_missing_values(ds)
    section_duplicate_questions(ds)
    section_perturbation_semantics(ds, alignment_info)

    emit("## Summary")
    emit()
    emit("See sections above for exact, run-grounded numbers. No values in this file were hand-written/fabricated; "
         "all were produced by executing this script against the on-disk dataset at the path listed at the top.")
    emit()

    FINDINGS_PATH.write_text("\n".join(REPORT_LINES), encoding="utf-8")
    print(f"\nFindings written to: {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
