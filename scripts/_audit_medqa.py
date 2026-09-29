"""
Forensic audit script for the locally-downloaded MedQA dataset.

Phase 0 — audit only. No API calls, no model training, no writes to
data/processed, no modification of the raw dataset.

Source: F:\\NAACL 2027\\data\\raw\\medqa  (datasets.load_from_disk -> DatasetDict
with splits train/validation/test). Originally downloaded from HF dataset
`premmahadik05/medqa`, config `questions`.

Output: F:\\NAACL 2027\\results\\audit\\_medqa_findings.md
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(r"F:\NAACL 2027")
DATA_PATH = ROOT / "data" / "raw" / "medqa"
OUT_PATH = ROOT / "results" / "audit" / "_medqa_findings.md"

REMEDQA_TARGET_N = 1259  # size of each medqa_* split in ReMedQA, for coarse comparison only


def normalize_text(s):
    """Lowercase, strip whitespace and punctuation for duplicate-matching."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = s.strip()
    return s


def is_missing(v):
    """True if value is None, empty string/list/dict, or all-whitespace string."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple, dict)) and len(v) == 0:
        return True
    return False


def fmt_val(v, maxlen=300):
    """Compact repr of a value for markdown embedding."""
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    s = str(s)
    if len(s) > maxlen:
        s = s[:maxlen] + f"... [truncated, full length={len(str(v))}]"
    return s


def main():
    lines = []
    lines.append("# MedQA Forensic Audit Findings\n")
    lines.append(
        "Source: `F:\\NAACL 2027\\data\\raw\\medqa` "
        "(loaded via `datasets.load_from_disk`). "
        "Originally from HF dataset `premmahadik05/medqa`, config `questions`.\n"
    )
    lines.append(
        "Phase 0 audit only. All numbers below are computed directly from the "
        "local on-disk data by this script; no statistic is fabricated. "
        "Anything not directly verifiable is marked `UNCERTAIN: <reason>`.\n"
    )

    print(f"Loading dataset from {DATA_PATH} ...")
    ds = load_from_disk(str(DATA_PATH))
    print("Loaded. Splits:", list(ds.keys()))

    split_names = list(ds.keys())
    lines.append(f"## Splits found\n\n`{split_names}`\n")

    # ------------------------------------------------------------------
    # Per-split schema / dtypes / n_examples
    # ------------------------------------------------------------------
    lines.append("## 1. Per-split schema, dtypes, and example counts\n")

    per_split_info = {}
    for split in split_names:
        d = ds[split]
        n = len(d)
        features = d.features
        col_info = []
        for col, feat in features.items():
            col_info.append(f"  - `{col}`: `{feat}`")
        per_split_info[split] = {"n": n, "columns": list(features.keys())}
        lines.append(f"### Split: `{split}`  (n_examples = {n})\n")
        lines.append("Columns / dtypes (as reported by `datasets` Features):\n")
        lines.extend(col_info)
        lines.append("")

    total_n = sum(v["n"] for v in per_split_info.values())
    lines.append(f"**Total examples across all splits: {total_n}**\n")

    # 1259 coarse signal
    lines.append("### Coarse size signal vs. ReMedQA `medqa_*` family (n=1259 per split)\n")
    for split in split_names:
        n = per_split_info[split]["n"]
        if n == REMEDQA_TARGET_N:
            note = f"EXACT MATCH to ReMedQA medqa_* split size ({REMEDQA_TARGET_N})."
        elif n > REMEDQA_TARGET_N:
            note = f"Superset-sized vs. {REMEDQA_TARGET_N} (larger by {n - REMEDQA_TARGET_N})."
        else:
            note = f"Subset-sized vs. {REMEDQA_TARGET_N} (smaller by {REMEDQA_TARGET_N - n})."
        lines.append(f"- `{split}`: n={n} -> {note}")
    lines.append(
        "\n(Raw counts only, reported for the overlap teammate's Task 5 analysis. "
        "No conclusion about shared provenance is drawn here.)\n"
    )

    # ------------------------------------------------------------------
    # 5 representative full examples per split
    # ------------------------------------------------------------------
    lines.append("## 2. Representative full examples (raw, unmodified)\n")
    for split in split_names:
        d = ds[split]
        n = len(d)
        lines.append(f"### `{split}` — 5 example records\n")
        if n == 0:
            lines.append("UNCERTAIN: split has 0 examples, nothing to show.\n")
            continue
        # spread indices across the split rather than just the first 5
        idxs = sorted(set([0] + [min(n - 1, (n * k) // 5) for k in range(1, 5)]))[:5]
        while len(idxs) < min(5, n):
            for cand in range(n):
                if cand not in idxs:
                    idxs.append(cand)
                    break
        idxs = idxs[:5]
        for i in idxs:
            ex = d[i]
            lines.append(f"**index {i}:**\n")
            lines.append("```json")
            lines.append(json.dumps(ex, ensure_ascii=False, indent=2))
            lines.append("```\n")

    # ------------------------------------------------------------------
    # Answer/label format
    # ------------------------------------------------------------------
    lines.append("## 3. Answer / label format\n")
    for split in split_names:
        d = ds[split]
        cols = d.column_names
        lines.append(f"### `{split}`\n")
        if "answer" in cols:
            sample_answers = [d[i]["answer"] for i in range(min(10, len(d)))]
            lines.append(f"- `answer` column present. Sample raw values: {fmt_val(sample_answers)}")
        else:
            lines.append("- `answer` column: UNCERTAIN — not present in this split's columns.")
        if "answer_idx" in cols:
            sample_idx = [d[i]["answer_idx"] for i in range(min(10, len(d)))]
            idx_value_counts = Counter(d["answer_idx"]) if len(d) else Counter()
            lines.append(f"- `answer_idx` column present. Sample raw values: {fmt_val(sample_idx)}")
            lines.append(f"  - Distinct `answer_idx` values and counts: {dict(idx_value_counts)}")
        else:
            lines.append("- `answer_idx` column: UNCERTAIN — not present in this split's columns.")
        lines.append("")

    lines.append(
        "**Interpretation:** `answer` stores the full option text (free-form string, "
        "matches the text of one of the `options` values); `answer_idx` stores the "
        "single-letter option key (e.g. `A`/`B`/`C`/`D`). This is verified directly "
        "against the sampled raw values above, not assumed.\n"
    )

    # verify answer text actually matches options[answer_idx] for a sample
    lines.append("### Cross-check: does `answer` text match `options[answer_idx]`?\n")
    for split in split_names:
        d = ds[split]
        cols = d.column_names
        if not ({"answer", "answer_idx", "options"} <= set(cols)):
            lines.append(f"- `{split}`: UNCERTAIN — missing one of answer/answer_idx/options columns.")
            continue
        n_check = min(200, len(d))
        matches = 0
        mismatches = []
        for i in range(n_check):
            ex = d[i]
            opts = ex["options"]
            idx = ex["answer_idx"]
            ans = ex["answer"]
            opt_val = opts.get(idx) if isinstance(opts, dict) else None
            if opt_val is not None and opt_val == ans:
                matches += 1
            else:
                if len(mismatches) < 2:
                    mismatches.append({"index": i, "answer": ans, "answer_idx": idx, "options": opts})
        lines.append(
            f"- `{split}`: {matches}/{n_check} sampled rows have `answer == options[answer_idx]`."
        )
        if mismatches:
            lines.append(f"  - Example mismatches: {fmt_val(mismatches, maxlen=500)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Options format
    # ------------------------------------------------------------------
    lines.append("## 4. Options format\n")
    for split in split_names:
        d = ds[split]
        cols = d.column_names
        if "options" not in cols:
            lines.append(f"- `{split}`: UNCERTAIN — no `options` column present.")
            continue
        n = len(d)
        key_counts = Counter()
        n_options_counts = Counter()
        storage_types = Counter()
        for i in range(n):
            opts = d[i]["options"]
            storage_types[type(opts).__name__] += 1
            if isinstance(opts, dict):
                key_counts[tuple(sorted(opts.keys()))] += 1
                n_options_counts[len(opts)] += 1
            elif isinstance(opts, list):
                n_options_counts[len(opts)] += 1
        lines.append(f"### `{split}`\n")
        lines.append(f"- Storage type of `options` field: {dict(storage_types)}")
        lines.append(f"- Distinct key-sets seen (dict form) and counts: {dict(key_counts)}")
        lines.append(f"- Distinct option-count values and counts: {dict(n_options_counts)}")
        lines.append("")
    lines.append(
        "**Interpretation:** options are stored as a fixed-key dict (`A`/`B`/`C`/`D`) "
        "per question, i.e. a fixed 4-way multiple choice format, unless the counts "
        "above show otherwise for a given split.\n"
    )

    # ------------------------------------------------------------------
    # id/uid column check
    # ------------------------------------------------------------------
    lines.append("## 5. Explicit id/uid column check\n")
    id_like_pattern = re.compile(r"(^id$|_id$|^uid$|_uid$|^uuid$|^qid$|^question_id$)", re.IGNORECASE)
    for split in split_names:
        d = ds[split]
        cols = d.column_names
        id_like = [c for c in cols if id_like_pattern.search(c)]
        lines.append(f"### `{split}`\n")
        lines.append(f"- Full column list: {cols}")
        if id_like:
            for c in id_like:
                samples = [d[i][c] for i in range(min(5, len(d)))]
                lines.append(f"- Found id-like column `{c}`. Sample values: {fmt_val(samples)}")
        else:
            lines.append(
                "- UNCERTAIN: no column matching id/uid/uuid/qid naming pattern found. "
                "No explicit identifier column exists in this split; row position "
                "(dataset index) is the only positional handle, and `meta_info` / "
                "`metamap_phrases` (if present) are NOT identifiers."
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Missing values check
    # ------------------------------------------------------------------
    lines.append("## 6. Missing value counts (question / options / answer / answer_idx)\n")
    for split in split_names:
        d = ds[split]
        n = len(d)
        cols = d.column_names
        lines.append(f"### `{split}` (n={n})\n")
        for col in ["question", "options", "answer", "answer_idx", "meta_info", "metamap_phrases"]:
            if col not in cols:
                lines.append(f"- `{col}`: UNCERTAIN — column not present in this split.")
                continue
            missing = 0
            missing_option_slots = 0  # for options: count of A/B/C/D sub-values that are empty
            for i in range(n):
                v = d[i][col]
                if col == "options" and isinstance(v, dict):
                    if len(v) == 0:
                        missing += 1
                    for sub_k, sub_v in v.items():
                        if is_missing(sub_v):
                            missing_option_slots += 1
                else:
                    if is_missing(v):
                        missing += 1
            extra = f" (missing individual option slots: {missing_option_slots})" if col == "options" else ""
            lines.append(f"- `{col}`: {missing} missing/empty/null out of {n}{extra}")
        lines.append("")

    # ------------------------------------------------------------------
    # Duplicate questions within and across splits
    # ------------------------------------------------------------------
    lines.append("## 7. Exact-duplicate question detection (normalized: lowercase, whitespace/punctuation stripped)\n")

    normalized_by_split = {}
    for split in split_names:
        d = ds[split]
        cols = d.column_names
        if "question" not in cols:
            normalized_by_split[split] = []
            lines.append(f"- `{split}`: UNCERTAIN — no `question` column present.")
            continue
        norm = [normalize_text(q) for q in d["question"]]
        normalized_by_split[split] = norm

    # within-split duplicates
    lines.append("### Within-split duplicates\n")
    within_dup_examples = {}
    for split in split_names:
        norm = normalized_by_split.get(split, [])
        if not norm:
            continue
        c = Counter(norm)
        dup_groups = {k: v for k, v in c.items() if v > 1 and k != ""}
        n_dup_examples = sum(v for v in dup_groups.values())  # total rows involved
        n_dup_groups = len(dup_groups)
        lines.append(
            f"- `{split}`: {n_dup_groups} distinct question(s) appear more than once, "
            f"accounting for {n_dup_examples} total rows (of {len(norm)})."
        )
        if dup_groups:
            example_key = next(iter(dup_groups))
            # find original (non-normalized) text + indices
            d = ds[split]
            idxs = [i for i, v in enumerate(norm) if v == example_key]
            within_dup_examples[split] = {
                "normalized_question": example_key,
                "indices": idxs[:5],
                "raw_examples": [d[i]["question"] for i in idxs[:2]],
            }
            lines.append(f"  - Example duplicate group (indices {idxs[:5]}): {fmt_val(within_dup_examples[split])}")
    lines.append("")

    # across-split duplicates
    lines.append("### Cross-split duplicates (train/validation/test overlap)\n")
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    for a, b in pairs:
        norm_a = normalized_by_split.get(a, [])
        norm_b = normalized_by_split.get(b, [])
        if not norm_a or not norm_b:
            lines.append(f"- `{a}` vs `{b}`: UNCERTAIN — one split missing `question` column or empty.")
            continue
        set_a = set(x for x in norm_a if x != "")
        set_b_counter = Counter(x for x in norm_b if x != "")
        overlap = set_a & set(set_b_counter.keys())
        n_overlap_distinct = len(overlap)
        # count total rows on each side involved in overlap
        a_counter = Counter(x for x in norm_a if x != "")
        rows_a_involved = sum(a_counter[x] for x in overlap)
        rows_b_involved = sum(set_b_counter[x] for x in overlap)
        lines.append(
            f"- `{a}` vs `{b}`: {n_overlap_distinct} distinct normalized question(s) appear in BOTH splits "
            f"({rows_a_involved} row(s) in `{a}`, {rows_b_involved} row(s) in `{b}`)."
        )
        if overlap:
            d_a = ds[a]
            d_b = ds[b]
            examples = []
            for k in list(overlap)[:2]:
                idx_a = norm_a.index(k)
                idx_b = norm_b.index(k)
                examples.append(
                    {
                        f"{a}_index": idx_a,
                        f"{a}_question": d_a[idx_a]["question"],
                        f"{b}_index": idx_b,
                        f"{b}_question": d_b[idx_b]["question"],
                    }
                )
            lines.append(f"  - Example overlap(s): {fmt_val(examples, maxlen=800)}")
            if (a, b) in [("train", "test"), ("train", "validation")]:
                lines.append(
                    f"  - **FLAG: potential train/{b} leakage** — {n_overlap_distinct} question(s) "
                    f"shared between `{a}` and `{b}`. This is a methodological concern worth "
                    f"surfacing loudly if the count is non-trivial."
                )
    lines.append("")

    # ------------------------------------------------------------------
    # Write findings
    # ------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFindings written to: {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
