"""
Build the ReMedQA pairing table (Phase 0, Task 8).

For every source (medqa/medmcqa/mmlu) and every shared id, joins the `mcq`
(original/baseline) row to each of the other 6 perturbation rows
(open, incorrect, roman_numeral, none_of_the_provided, fixed_pos, no_symbols)
sharing that id, producing a long-format table with one row per
(source, id, perturbation_type) pair.

Pairing basis: id-based join on `{source}_mcq` vs each `{source}_{perturbation}`.
This is verified reliable by the teammate's forensic audit
(results/audit/_remedqa_findings.md, Section 3): for every source, the id set
is IDENTICAL across all 7 perturbation splits (intersection == union, zero
gaps, zero duplicate ids within any split). This script re-verifies that
claim programmatically before building the table (it does not blindly trust
the audit doc), and will hard-fail loudly (not silently drop rows) if the
claim does not hold for the on-disk data at run time.

Does NOT modify data/raw. Writes only to data/processed/.

Run: uv run python scripts\\build_pairs.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "remedqa"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PARQUET = PROCESSED_DIR / "remedqa_pairs.parquet"
DOC_PATH = PROCESSED_DIR / "PAIRING_METHOD.md"

SOURCES = ["medqa", "medmcqa", "mmlu"]
BASELINE = "mcq"
PERTURBATIONS = [
    "open",
    "incorrect",
    "roman_numeral",
    "none_of_the_provided",
    "fixed_pos",
    "no_symbols",
]
ALL_PERTURBATIONS = [BASELINE] + PERTURBATIONS

# Perturbations whose `answer` field is already free text comparable to
# original_answer_text (per teammate's verified perturbation semantics).
TEXT_ANSWER_PERTURBATIONS = {"open", "no_symbols"}
# Perturbations whose `answer` field is a letter/roman-numeral key into
# `perturbed_options` that must be resolved to text.
KEY_ANSWER_PERTURBATIONS = {"roman_numeral", "none_of_the_provided", "fixed_pos"}
# The task-inverted perturbation: answer is a LIST of wrong keys, not a
# correctness target at all.
INVERTED_PERTURBATIONS = {"incorrect"}


def parse_options_field(raw: Any) -> Any:
    """Parse ReMedQA's `options` field, which is always a Python dict-repr
    string (e.g. "{'A': '...', 'B': '...'}"), NOT valid JSON. Use
    ast.literal_eval, not json.loads (json.loads can silently misparse
    values inside, per verified audit finding).
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    return ast.literal_eval(raw)


def parse_answer_field(raw: Any) -> Any:
    """Parse ReMedQA's `answer` field.

    Unlike `options`, `answer` is NOT uniformly a Python-literal repr: for
    most perturbations it is a bare plain-text string (e.g. "B", "II",
    "Cross-linking of DNA", "0.01") which is NOT valid input to
    ast.literal_eval (a bare unquoted letter/word parses as an ast.Name,
    and free text with spaces is not a valid Python expression at all).
    Only the `incorrect` perturbation's answer is an actual Python literal:
    a list-repr string like "['A', 'C', 'D']".

    So: only attempt ast.literal_eval when the raw string looks like a
    list-repr (starts with '['); otherwise treat the raw string itself as
    the parsed value (this matches the verified audit findings — mcq
    answer is a bare letter, roman_numeral answer is a bare roman numeral,
    open/no_symbols answer is bare free text of the correct option).
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return ast.literal_eval(raw)
    return raw


def normalize_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    out = s.lower().strip()
    out = "".join(ch for ch in out if ch.isalnum() or ch.isspace())
    return " ".join(out.split())


def load_dataset() -> DatasetDict:
    ds = load_from_disk(str(RAW_DATA_DIR))
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {RAW_DATA_DIR}, got {type(ds).__name__}")
    return ds


def verify_id_alignment(ds: DatasetDict) -> dict[str, list[str]]:
    """Re-verify (do not just trust) that id sets are identical across all 7
    perturbations for each source. Returns {source: sorted list of ids}.
    Raises if the claim does not hold.
    """
    source_ids: dict[str, list[str]] = {}
    for source in SOURCES:
        id_sets = {}
        for pert in ALL_PERTURBATIONS:
            split_name = f"{source}_{pert}"
            if split_name not in ds:
                raise RuntimeError(f"Missing expected split: {split_name}")
            ids = ds[split_name]["id"]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate ids found within split {split_name}")
            id_sets[pert] = set(ids)

        union = set().union(*id_sets.values())
        intersection = set.intersection(*id_sets.values())
        if union != intersection:
            raise RuntimeError(
                f"Source {source}: id sets differ across perturbations "
                f"(union={len(union)}, intersection={len(intersection)}). "
                "Pairing assumption violated — aborting rather than silently "
                "producing an incomplete table."
            )
        source_ids[source] = sorted(intersection)
        print(f"[verify] {source}: {len(intersection)} ids identical across all 7 perturbation splits.")
    return source_ids


def build_row_lookup(ds: DatasetDict, source: str, pert: str) -> dict[str, dict]:
    split = ds[f"{source}_{pert}"]
    return {row["id"]: row for row in split}


def resolve_letter_to_text(options: dict, letter: str) -> Any:
    if options is None or letter is None:
        return None
    return options.get(letter)


def build_pairs(ds: DatasetDict, source_ids: dict[str, list[str]]) -> pd.DataFrame:
    records: list[dict] = []

    for source in SOURCES:
        ids = source_ids[source]
        # Build id->row lookups for all 7 perturbations of this source once.
        lookups = {pert: build_row_lookup(ds, source, pert) for pert in ALL_PERTURBATIONS}

        for source_id in ids:
            mcq_row = lookups[BASELINE][source_id]
            orig_question = mcq_row["question"]
            orig_options = parse_options_field(mcq_row["options"])
            orig_answer_letter = parse_answer_field(mcq_row["answer"])
            orig_answer_text = resolve_letter_to_text(orig_options, orig_answer_letter)

            for pert in PERTURBATIONS:
                row = lookups[pert][source_id]
                perturbed_question = row["question"]
                perturbed_options_raw = parse_options_field(row["options"])
                perturbed_answer_raw_parsed = parse_answer_field(row["answer"])

                is_task_inverted = pert in INVERTED_PERTURBATIONS

                if pert in TEXT_ANSWER_PERTURBATIONS:
                    # open, no_symbols: answer field is already the correct
                    # option's text.
                    perturbed_answer_interpretation = perturbed_answer_raw_parsed
                elif pert in KEY_ANSWER_PERTURBATIONS:
                    # roman_numeral, none_of_the_provided, fixed_pos: answer
                    # is a key into perturbed_options; resolve to text so
                    # it's comparable to original_answer_text.
                    perturbed_answer_interpretation = resolve_letter_to_text(
                        perturbed_options_raw, perturbed_answer_raw_parsed
                    )
                elif pert in INVERTED_PERTURBATIONS:
                    # incorrect: answer is a LIST of wrong-answer letters —
                    # this INVERTS the task. Do NOT treat as a correctness
                    # target. Set interpretation to None; flag via
                    # is_task_inverted instead.
                    perturbed_answer_interpretation = None
                else:
                    raise RuntimeError(f"Unhandled perturbation type: {pert}")

                question_text_changed = normalize_text(perturbed_question) != normalize_text(orig_question)

                metadata = {
                    "original_prompt": mcq_row.get("prompt"),
                    "original_prompt_think": mcq_row.get("prompt_think"),
                    "perturbed_prompt": row.get("prompt"),
                    "perturbed_prompt_think": row.get("prompt_think"),
                    "perturbed_answer_raw_field": row["answer"],  # unparsed string as stored
                }

                records.append(
                    {
                        "source_dataset": source,
                        "source_id": source_id,
                        "perturbation_type": pert,
                        "original_question": orig_question,
                        "original_options": json.dumps(orig_options, ensure_ascii=False),
                        "original_answer_letter": orig_answer_letter,
                        "original_answer_text": orig_answer_text,
                        "perturbed_question": perturbed_question,
                        "perturbed_options": (
                            json.dumps(perturbed_options_raw, ensure_ascii=False)
                            if perturbed_options_raw is not None
                            else None
                        ),
                        "perturbed_answer_raw": row["answer"],
                        "perturbed_answer_interpretation": perturbed_answer_interpretation,
                        "is_task_inverted": is_task_inverted,
                        "question_text_changed": question_text_changed,
                        "metadata": json.dumps(metadata, ensure_ascii=False),
                    }
                )

    return pd.DataFrame.from_records(records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    source_ids = verify_id_alignment(ds)

    n_originals = sum(len(v) for v in source_ids.values())
    expected_rows = n_originals * len(PERTURBATIONS)
    print(f"[info] originals (source x id): {n_originals}")
    print(f"[info] expected pairing rows ({n_originals} x {len(PERTURBATIONS)} perturbations): {expected_rows}")

    df = build_pairs(ds, source_ids)
    print(f"[info] actual pairing rows produced: {len(df)}")

    if len(df) != expected_rows:
        print(
            f"WARNING: actual row count {len(df)} != expected {expected_rows}. "
            "Reporting the real number, not forcing it."
        )

    # Sanity checks
    assert df["source_id"].notna().all()
    assert df["perturbation_type"].nunique() == len(PERTURBATIONS)
    per_source_counts = df.groupby("source_dataset")["source_id"].nunique().to_dict()
    print(f"[info] distinct originals per source in output: {per_source_counts}")

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"[info] wrote {OUTPUT_PARQUET} ({len(df)} rows, {len(df.columns)} columns)")

    write_companion_doc(n_originals, expected_rows, len(df), per_source_counts)
    print(f"[info] wrote {DOC_PATH}")


def write_companion_doc(n_originals: int, expected_rows: int, actual_rows: int, per_source_counts: dict) -> None:
    doc = f"""# ReMedQA Pairing Table — Method

Produced by `scripts/build_pairs.py` from `data/raw/remedqa`.
Output: `data/processed/remedqa_pairs.parquet` ({actual_rows} rows).

## How pairing was established

Pairing is **id-based**, not positional/index-based. For each source
(medqa/medmcqa/mmlu), each row of the `{{source}}_mcq` split (the natural
baseline — standard MCQ format, letter-labeled options, letter answer) is
joined to the row with the SAME `id` in each of the other 6 perturbation
splits: `open`, `incorrect`, `roman_numeral`, `none_of_the_provided`,
`fixed_pos`, `no_symbols`.

This is verified reliable both by the teammate's forensic audit
(`results/audit/_remedqa_findings.md`, Section 3 — id set is IDENTICAL
across all 7 perturbation splits per source, intersection == union, zero
gaps, zero duplicate ids within any split) AND re-verified programmatically
by `build_pairs.py` itself at run time (`verify_id_alignment()`), which
hard-fails rather than silently dropping rows if the assumption doesn't
hold for the on-disk data.

Per-source originals joined: {per_source_counts}
Total originals (source, id) pairs: {n_originals}
Expected pairing rows ({n_originals} originals x 6 non-mcq perturbations): {expected_rows}
Actual pairing rows produced: {actual_rows}

Each output row = one (source_dataset, source_id, perturbation_type) triple,
i.e. one original-question-vs-one-perturbed-variant comparison. There are 6
rows per original question (one per non-mcq perturbation).

## Schema

- `source_dataset`: medqa / medmcqa / mmlu
- `source_id`: the shared id joining the mcq row to the perturbed row
- `perturbation_type`: one of open, incorrect, roman_numeral,
  none_of_the_provided, fixed_pos, no_symbols
- `original_question` / `original_options` (JSON string of parsed dict) /
  `original_answer_letter` / `original_answer_text`: from the `{{source}}_mcq`
  baseline row for this id.
- `perturbed_question` / `perturbed_options` (JSON string of parsed dict,
  present for ALL 6 perturbations here — on inspection every perturbation
  split, including `open`, still carries an `options` field with the same
  A/B/C/D-keyed dict as the baseline; it is not null for any perturbation in
  this dataset) / `perturbed_answer_raw` (the raw, unparsed `answer` string
  exactly as stored for that perturbation).
- `perturbed_answer_interpretation`: a NORMALIZED field computed by this
  script so it is comparable to `original_answer_text`:
  - `open`, `no_symbols`: pass-through of the raw answer text (already free
    text of the correct option).
  - `roman_numeral`, `none_of_the_provided`, `fixed_pos`: the raw
    letter/roman-numeral key resolved to its option TEXT via that row's own
    `perturbed_options` dict (each perturbation's own key scheme).
  - `incorrect`: **set to None/null**. See caveat below.
- `is_task_inverted` (bool): True only for `perturbation_type == "incorrect"`.
- `question_text_changed` (bool): True if `perturbed_question`, after
  lowercasing/punctuation-stripping/whitespace-collapsing normalization,
  differs from `original_question` under the same normalization. This flags
  which perturbations actually reworded the QUESTION text (e.g. `open`) vs.
  which only changed the OPTIONS/answer-key scheme while leaving the
  question itself unchanged (e.g. `roman_numeral`, `fixed_pos`,
  `none_of_the_provided`, `no_symbols` are expected to leave the question
  text alone based on the teammate's audit; `incorrect` also does not reword
  the question — it only changes what's being asked of the answer).
  `question_text_changed` is about the `question` STRING field only; it
  says nothing about the `incorrect` perturbation's task-level inversion
  (see below), which is a semantic change even though the printed question
  text is unchanged.
- `metadata`: JSON string containing the original and perturbed `prompt` /
  `prompt_think` fields (verbatim from source) plus the raw unparsed
  `answer` field, for anyone downstream who wants the full prompt context.

## IMPORTANT — `incorrect` perturbation task-inversion caveat

For `perturbation_type == "incorrect"`, the raw `answer` field is a LIST of
the WRONG option letters (e.g. `['A', 'C', 'D']`), not the correct answer.
This perturbation INVERTS the task: it asks the model to identify the
incorrect options, not to answer the original question correctly. It is
**not** a meaning-preserving rephrasing of the same question/answer pair.

Accordingly:
- `perturbed_answer_interpretation` is set to `None` for all `incorrect`
  rows — do NOT treat it as a correctness target, and do NOT compare it
  directly to `original_answer_text`.
- `is_task_inverted = True` flags every `incorrect` row so downstream
  consumers can filter/handle it separately (e.g. exclude from any
  "did perturbation X preserve the correct answer" analysis, or score it
  against the ORIGINAL question's incorrect-option set instead, which is a
  different, deliberate task).
- `perturbed_answer_raw` for `incorrect` rows still contains the raw
  Python-repr list string exactly as stored (e.g. `"['A', 'C', 'D']"`), for
  anyone who explicitly wants to work with the inverted task.

## `question_text_changed` semantics — do not misuse

This column is a purely lexical/string diff signal (after light
normalization), computed once per pairing row. It answers "did the visible
question text change" — it is NOT a semantic-equivalence judgment and is
NOT a proxy for whether the perturbation changed the underlying task. In
particular:
- `open` is expected to have `question_text_changed = True` often (question
  is reworded to be self-contained / drop "which of the following").
- `incorrect` is expected to have `question_text_changed = False` in most
  cases (the question string is typically unchanged) even though the TASK
  itself is inverted (see caveat above) — a downstream user must consult
  `is_task_inverted`, not `question_text_changed`, to detect the inversion.
- Real per-perturbation rates are reported by
  `results/audit/perturbation_descriptives.csv` (Task 9), not asserted here.

## Raw-string parsing note

`options` and `answer` fields in the raw dataset are Python-repr strings
(e.g. `"{{'A': '...', 'B': '...'}}"`), NOT valid JSON. This script parses them
with `ast.literal_eval`, not `json.loads` (per verified audit finding:
`json.loads` silently misparses some numeric-looking answer strings, e.g.
`"0.01"`, as floats rather than leaving them as the intended string/label).
`original_options` and `perturbed_options` are stored in the output parquet
as JSON strings (via `json.dumps` after `ast.literal_eval` parsing) for
portability across parquet readers; downstream consumers should
`json.loads()` them to get back a dict.
"""
    DOC_PATH.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
