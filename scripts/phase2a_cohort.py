"""Deterministic Phase 2A cohort: 50 underlying questions drawn ONLY from the
fixed pilot_200 cohort (never the full dataset), stratified by source in
proportion to the pilot's own composition, seed 42 (same as pilot).

Run: uv run python scripts\\phase2a_cohort.py
"""

import hashlib
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "processed" / "pilot_200.parquet"
OUT = ROOT / "data" / "processed" / "phase2a_50_manifest.csv"
N_TARGET = 50
SEED = 42


def largest_remainder(counts: dict, total: int) -> dict:
    whole = sum(counts.values())
    raw = {k: v * total / whole for k, v in counts.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - alloc[k], k), reverse=True)[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main():
    pilot = pd.read_parquet(PILOT)
    ids = pilot[["source_dataset", "source_id"]].drop_duplicates()
    assert len(ids) == 200, f"pilot should contain 200 unique questions, found {len(ids)}"
    counts = ids.groupby("source_dataset").size().to_dict()
    alloc = largest_remainder(counts, N_TARGET)

    rng = random.Random(SEED)
    selected = {}
    for src in sorted(alloc):
        pool = sorted(ids[ids["source_dataset"] == src]["source_id"].tolist())
        selected[src] = sorted(rng.sample(pool, alloc[src]))

    # Round-robin stage order so the 5-question Stage 2 spans all sources.
    order, srcs = [], ["medqa", "medmcqa", "mmlu"]
    i = 0
    while len(order) < N_TARGET:
        for s in srcs:
            if i < len(selected[s]):
                order.append((s, selected[s][i]))
        i += 1

    rows = [{
        "stage_order": k + 1,
        "question_id": f"{s}::{sid}",
        "source_dataset": s,
        "source_id": sid,
        "pilot_cohort_seed": 42,
        "phase2a_selection_seed": SEED,
        "selection_method": f"stratified largest-remainder {alloc}, random.Random({SEED}).sample over sorted pilot ids per source",
    } for k, (s, sid) in enumerate(order)]
    df = pd.DataFrame(rows)
    assert df["question_id"].is_unique and len(df) == N_TARGET
    df.to_csv(OUT, index=False)

    digest = hashlib.sha256("\n".join(df["question_id"]).encode()).hexdigest()
    print(f"Allocation: {alloc}")
    print(f"Wrote {OUT} ({len(df)} questions), cohort sha256={digest}")


if __name__ == "__main__":
    main()
