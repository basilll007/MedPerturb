"""
Build the deterministic ReMedQA pilot cohort (Phase 0, Task 10).

Samples ~200 ORIGINAL ReMedQA questions (distinct (source_dataset, source_id)
pairs from the `mcq` baseline) using proportional stratified sampling by
source, then pulls in ALL of each sampled original's 6 perturbed variants
from the pairing table (data/processed/remedqa_pairs.parquet). This does
NOT randomly sample 200 unrelated perturbed rows — every sampled original
keeps its full set of linked perturbations.

Why stratified (not pooled uniform) sampling: the 3 sources have very
different pool sizes (medqa=1259, medmcqa=1000, mmlu=895; total 3154).
Pure uniform sampling across the pooled 3154 ids would, in expectation,
still land close to proportional (since it's a large pool), but stratified
sampling GUARANTEES each source is represented in proportion to its true
size in every run, removes sampling variance in the source mix, and makes
the per-source pilot sizes an explicit, auditable, documented decision
rather than an incidental outcome of one random draw. This matters here
because downstream pilot-scale human/LLM-judge annotation work will likely
want known, stable per-source counts to reason about coverage.

Random source: `random.Random(42)` (Python's own PRNG, seeded), used via
`.sample()` for each source's id list independently, in source order
[medqa, medmcqa, mmlu]. (Not numpy — documented explicitly here so nobody
has to guess which generator produced these ids.)

Run: uv run python scripts\\build_pilot.py
Reads:  data/processed/remedqa_pairs.parquet
Writes: data/processed/pilot_200.parquet
        data/processed/PILOT_COHORT.md (companion note)
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAIRS_PARQUET = PROJECT_ROOT / "data" / "processed" / "remedqa_pairs.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PILOT_PARQUET = PROCESSED_DIR / "pilot_200.parquet"
DOC_PATH = PROCESSED_DIR / "PILOT_COHORT.md"

SEED = 42
TARGET_TOTAL = 200
SOURCES = ["medqa", "medmcqa", "mmlu"]


def compute_stratified_targets(pool_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    """Proportional allocation of target_total across sources by pool size,
    using largest-remainder rounding so the per-source targets sum exactly
    to target_total (or as close as integer allocation allows)."""
    total_pool = sum(pool_sizes.values())
    raw = {s: target_total * pool_sizes[s] / total_pool for s in SOURCES}
    floor_alloc = {s: int(raw[s]) for s in SOURCES}
    allocated = sum(floor_alloc.values())
    remainder = target_total - allocated
    # Distribute the remaining seats to the sources with the largest
    # fractional remainder (largest-remainder / Hamilton method).
    remainders = sorted(SOURCES, key=lambda s: raw[s] - floor_alloc[s], reverse=True)
    for i in range(remainder):
        floor_alloc[remainders[i % len(SOURCES)]] += 1
    return floor_alloc


def main() -> None:
    pairs = pd.read_parquet(PAIRS_PARQUET)
    print(f"[info] loaded {len(pairs)} pairing rows from {PAIRS_PARQUET}")

    originals = pairs[["source_dataset", "source_id"]].drop_duplicates().reset_index(drop=True)
    pool_sizes = originals.groupby("source_dataset")["source_id"].nunique().to_dict()
    pool_sizes = {s: pool_sizes[s] for s in SOURCES}
    print(f"[info] pool sizes per source: {pool_sizes}")

    targets = compute_stratified_targets(pool_sizes, TARGET_TOTAL)
    print(f"[info] stratified per-source targets (SEED={SEED}, target_total={TARGET_TOTAL}): {targets}")

    rng = random.Random(SEED)
    sampled_ids: dict[str, list[str]] = {}
    for source in SOURCES:
        # Sort first for determinism (dict/set ordering from upstream
        # operations is not guaranteed stable across pandas versions), then
        # sample with Python's seeded random.Random.
        pool = sorted(originals.loc[originals["source_dataset"] == source, "source_id"].tolist())
        k = min(targets[source], len(pool))
        sampled_ids[source] = rng.sample(pool, k)
        print(f"[info] sampled {len(sampled_ids[source])} / {len(pool)} ids for source={source}")

    total_sampled = sum(len(v) for v in sampled_ids.values())
    print(f"[info] total sampled originals: {total_sampled} (target was {TARGET_TOTAL})")

    keep_mask = pd.Series(False, index=pairs.index)
    for source in SOURCES:
        id_set = set(sampled_ids[source])
        keep_mask |= (pairs["source_dataset"] == source) & (pairs["source_id"].isin(id_set))

    pilot_df = pairs.loc[keep_mask].copy()
    pilot_df["pilot_cohort"] = True

    expected_rows = total_sampled * 6  # 6 non-mcq perturbations per original
    print(f"[info] pilot pairing rows: {len(pilot_df)} (expected {expected_rows} = {total_sampled} originals x 6 perturbations)")
    assert len(pilot_df) == expected_rows, "Pilot row count does not match sampled originals x 6 perturbations"

    per_source_counts = pilot_df.groupby("source_dataset")["source_id"].nunique().to_dict()
    print(f"[info] distinct pilot originals per source (final): {per_source_counts}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pilot_df.to_parquet(PILOT_PARQUET, index=False)
    print(f"[info] wrote {PILOT_PARQUET} ({len(pilot_df)} rows, {len(pilot_df.columns)} columns)")

    write_companion_doc(pool_sizes, targets, sampled_ids, total_sampled, len(pilot_df), per_source_counts)
    print(f"[info] wrote {DOC_PATH}")


def write_companion_doc(pool_sizes, targets, sampled_ids, total_sampled, total_rows, per_source_counts) -> None:
    per_source_lines = "\n".join(
        f"- {s}: pool={pool_sizes[s]}, proportional target={targets[s]}, "
        f"actually sampled={len(sampled_ids[s])}"
        for s in SOURCES
    )
    doc = f"""# ReMedQA Pilot Cohort (SEED={SEED}) — Method & Schema

Produced by `scripts/build_pilot.py` from
`data/processed/remedqa_pairs.parquet`.
Output: `data/processed/pilot_200.parquet` ({total_rows} rows).

## Sampling method

- Random source: Python's `random.Random({SEED})` — NOT numpy. Each source's
  id pool is sorted (for determinism) and then sampled independently via
  `.sample(pool, k)`, in source order [medqa, medmcqa, mmlu].
- Target: approximately {TARGET_TOTAL} distinct ORIGINAL (source_dataset,
  source_id) pairs from the `mcq` baseline, i.e. {TARGET_TOTAL} questions,
  each with all 6 of its linked non-mcq perturbations pulled in from
  `remedqa_pairs.parquet` (NOT 200 randomly sampled unrelated perturbed
  rows — every sampled original keeps its full perturbation family).
- Stratification: PROPORTIONAL stratified sampling by source, not pooled
  uniform sampling across all 3154 originals. Per-source target counts are
  computed as `{TARGET_TOTAL} * pool_size[source] / total_pool_size`, then
  rounded to integers via the largest-remainder (Hamilton) method so the
  per-source targets sum to exactly {TARGET_TOTAL} (subject to final
  reporting of the true achieved total below, since source pools sample
  without replacement and the rounding is not forced further than this).

### Why stratified rather than pooled uniform sampling

The 3 sources have very different pool sizes ({pool_sizes}). Uniform random
sampling across the pooled {sum(pool_sizes.values())} originals would, in
expectation, land close to the same proportions purely because the pools
are large — but "close in expectation" is not the same as "guaranteed and
auditable." Stratified sampling:
1. Guarantees every source is represented in the pilot in proportion to its
   true size in THIS run, not just on average across hypothetical reruns.
2. Removes sampling variance in the source mix, which matters for a small
   n≈200 pilot where a single draw's imbalance could otherwise
   meaningfully skew per-source pilot analysis.
3. Makes the per-source pilot sizes an explicit, documented, reproducible
   decision rather than an incidental byproduct of one random draw.

### Per-source target counts and rounding

{per_source_lines}

Total originals actually sampled: {total_sampled} (target was {TARGET_TOTAL}).
Rounding via largest-remainder means the true total may land slightly off
{TARGET_TOTAL} — this is expected and is not forced back to exactly
{TARGET_TOTAL}; the real achieved number is reported above and in the
script's stdout log.

## Output schema

`pilot_200.parquet` reuses the exact row schema of
`data/processed/remedqa_pairs.parquet` (see
`data/processed/PAIRING_METHOD.md` for the full column-by-column
definition), filtered down to only the rows whose `(source_dataset,
source_id)` is one of the {total_sampled} sampled originals, PLUS one
additional column:

- `pilot_cohort` (bool): always `True` in this file (every row in
  `pilot_200.parquet` is, by construction, part of the pilot cohort — the
  column exists so this file can be concatenated with, or diffed against,
  the full `remedqa_pairs.parquet` without losing the provenance flag, or
  so a filtered copy re-joined into the full table downstream stays
  self-describing).

Row count: {total_rows} = {total_sampled} sampled originals x 6 non-mcq
perturbations each (open, incorrect, roman_numeral, none_of_the_provided,
fixed_pos, no_symbols). The `incorrect` perturbation's task-inversion
caveat and `question_text_changed` semantics documented in
`PAIRING_METHOD.md` apply identically here — this file does not redefine
or relax them.

## Reproducibility

Given the same `remedqa_pairs.parquet` (itself deterministically derived
from the raw dataset), re-running `scripts/build_pilot.py` with
`SEED={SEED}` will reproduce the exact same sampled id sets, because the
per-source id pools are sorted before sampling and `random.Random({SEED})`
is deterministic.
"""
    DOC_PATH.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
