"""Leakage-safe grouped dataset splitting for Phase 3.

Splits ReMedQA original questions into four distinct, leakage-safe groups:
1. `train`: previously unexamined questions for post-training (LoRA/GRPO).
2. `val`: previously unexamined validation questions for model selection and checkpointing.
3. `diagnostic`: contains ALL previously inspected questions (the 200-question pilot cohort
   and the 50-question Phase 2A subset, plus any questions clustered with them by identical stems).
   Used for sanity checks, sanity evaluation, and direct comparison against Phase 2A Gemini baselines.
4. `final_test`: strictly untouched, previously unexamined held-out questions. Evaluated ONLY
   after training and hyperparameters are completely frozen.

Guarantees:
- Every underlying question (and all 7 of its representations: mcq baseline + 6 perturbations)
  stays strictly within a single split (grouped by question_id).
- Questions sharing identical normalized question text (transitive equivalence classes via graph
  connected components) are assigned together to the same split, preventing question-stem leakage.
- Proportional stratification by source_dataset (medqa, medmcqa, mmlu) is preserved.
- Deterministic given seed=42.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

SPLIT_NAMES = ["train", "val", "diagnostic", "final_test"]


def normalize_question_text(text: str) -> str:
    """Normalize question text matching the Phase 0 forensic audit definition:
    lowercase -> strip non-alphanumeric punctuation -> collapse whitespace runs.
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_question_clusters(
    pairs_df: pd.DataFrame,
    pilot_qids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build connected components (clusters) of questions that must never be separated.

    Nodes are question_ids and normalized question texts. An edge connects a question_id
    to its normalized question text. The connected components define the atomic clusters
    that must be kept together.
    """
    pilot_qids = pilot_qids or set()
    mcq_df = pairs_df[["source_dataset", "source_id", "original_question"]].drop_duplicates().copy()
    mcq_df["norm_q"] = mcq_df["original_question"].map(normalize_question_text)
    mcq_df["question_id"] = mcq_df["source_dataset"] + "::" + mcq_df["source_id"]

    g = nx.Graph()
    for _, r in mcq_df.iterrows():
        qid = r["question_id"]
        nq_node = f"NORM::{r['norm_q']}"
        g.add_edge(qid, nq_node)

    components = sorted(
        list(nx.connected_components(g)),
        key=lambda comp: sorted([x for x in comp if not x.startswith("NORM::")])[0],
    )
    clusters = []

    for i, comp in enumerate(components):
        qids = sorted([x for x in comp if not x.startswith("NORM::")])
        norm_texts = sorted([x[6:] for x in comp if x.startswith("NORM::")])
        sources = {q.split("::")[0] for q in qids}
        if len(sources) > 1:
            raise ValueError(f"Cluster {i} spans multiple source datasets: {sources}")
        source_dataset = next(iter(sources))

        has_pilot = any(q in pilot_qids for q in qids)

        clusters.append({
            "cluster_id": f"cluster_{i:04d}",
            "source_dataset": source_dataset,
            "question_ids": qids,
            "normalized_texts": norm_texts,
            "size": len(qids),
            "has_pilot": has_pilot,
        })

    return clusters


def split_clusters_four_way(
    clusters: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each cluster to 'train', 'val', 'diagnostic', or 'final_test'.

    - All clusters containing any pilot question (has_pilot=True) go to 'diagnostic'.
    - Non-pilot clusters are stratified by source into 'train', 'val', and 'final_test'.
    """
    rng = random.Random(seed)
    split_assignment: dict[str, str] = {}

    sources = sorted(list({c["source_dataset"] for c in clusters}))

    for src in sources:
        src_clusters = [c for c in clusters if c["source_dataset"] == src]
        pilot_c = [c for c in src_clusters if c["has_pilot"]]
        non_pilot_c = [c for c in src_clusters if not c["has_pilot"]]

        # 1. Pilot clusters strictly assigned to diagnostic
        for c in pilot_c:
            split_assignment[c["cluster_id"]] = "diagnostic"

        # 2. Non-pilot clusters partitioned deterministically
        non_pilot_c = sorted(non_pilot_c, key=lambda c: c["question_ids"][0])
        rng.shuffle(non_pilot_c)

        total_q = sum(c["size"] for c in src_clusters)
        target_train_q = int(round(total_q * train_ratio))
        target_val_q = int(round(total_q * val_ratio))

        curr_train = 0
        curr_val = 0

        for c in non_pilot_c:
            if curr_train < target_train_q:
                split_assignment[c["cluster_id"]] = "train"
                curr_train += c["size"]
            elif curr_val < target_val_q:
                split_assignment[c["cluster_id"]] = "val"
                curr_val += c["size"]
            else:
                split_assignment[c["cluster_id"]] = "final_test"

    return split_assignment


def verify_split_leakage(
    manifest_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    pilot_qids: set[str] | None = None,
    phase2a_qids: set[str] | None = None,
) -> list[str]:
    """Exhaustively verify that no leakage exists among train, val, diagnostic, and final_test splits.

    Returns a list of error strings. If clean, returns an empty list.
    """
    errors = []

    splits = SPLIT_NAMES
    qids_by_split: dict[str, set[str]] = {
        s: set(manifest_df.loc[manifest_df["split"] == s, "question_id"]) for s in splits
    }

    # 1. Question ID Disjointness
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            intersection = qids_by_split[s1] & qids_by_split[s2]
            if intersection:
                errors.append(
                    f"Question ID leakage between {s1} and {s2}: {len(intersection)} overlapping IDs: {sorted(list(intersection))[:5]}"
                )

    # 2. Normalized Question Text Disjointness
    mcq_df = pairs_df[["source_dataset", "source_id", "original_question"]].drop_duplicates().copy()
    mcq_df["question_id"] = mcq_df["source_dataset"] + "::" + mcq_df["source_id"]
    mcq_df["norm_q"] = mcq_df["original_question"].map(normalize_question_text)
    q_to_norm = dict(zip(mcq_df["question_id"], mcq_df["norm_q"]))

    norms_by_split: dict[str, set[str]] = {
        s: {q_to_norm[q] for q in qids_by_split[s] if q in q_to_norm} for s in splits
    }

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            intersection = norms_by_split[s1] & norms_by_split[s2]
            if intersection:
                errors.append(
                    f"Normalized text leakage between {s1} and {s2}: {len(intersection)} shared stems: {sorted(list(intersection))[:5]}"
                )

    # 3. Pilot Cohort Containment: pilot questions must be in diagnostic only
    if pilot_qids:
        outside_diagnostic_pilot = (qids_by_split["train"] | qids_by_split["val"] | qids_by_split["final_test"]) & pilot_qids
        if outside_diagnostic_pilot:
            errors.append(
                f"Pilot cohort leaked outside diagnostic: {len(outside_diagnostic_pilot)} IDs: {sorted(list(outside_diagnostic_pilot))[:5]}"
            )
        diag_pilot_count = len(qids_by_split["diagnostic"] & pilot_qids)
        if diag_pilot_count != len(pilot_qids):
            errors.append(f"Expected all {len(pilot_qids)} pilot questions in diagnostic, got {diag_pilot_count}")

    # 4. Phase 2A Cohort Containment: Phase 2A questions must be in diagnostic only
    if phase2a_qids:
        outside_diagnostic_p2a = (qids_by_split["train"] | qids_by_split["val"] | qids_by_split["final_test"]) & phase2a_qids
        if outside_diagnostic_p2a:
            errors.append(
                f"Phase 2A cohort leaked outside diagnostic: {len(outside_diagnostic_p2a)} IDs: {sorted(list(outside_diagnostic_p2a))[:5]}"
            )
        diag_p2a_count = len(qids_by_split["diagnostic"] & phase2a_qids)
        if diag_p2a_count != len(phase2a_qids):
            errors.append(f"Expected all {len(phase2a_qids)} Phase 2A questions in diagnostic, got {diag_p2a_count}")

    # 5. Row Completeness across conditions
    all_qids = set(manifest_df["question_id"])
    pairs_qids = set(pairs_df["source_dataset"] + "::" + pairs_df["source_id"])
    if all_qids != pairs_qids:
        errors.append(f"Manifest questions ({len(all_qids)}) do not match pairs table questions ({len(pairs_qids)})")

    counts_per_q = Counter(pairs_df["source_dataset"] + "::" + pairs_df["source_id"])
    bad_counts = {q: cnt for q, cnt in counts_per_q.items() if cnt != 6}
    if bad_counts:
        errors.append(f"{len(bad_counts)} questions have != 6 perturbation rows in pairs table: {list(bad_counts.items())[:5]}")

    invalid_splits = manifest_df[~manifest_df["split"].isin(splits)]
    if len(invalid_splits) > 0:
        errors.append(f"{len(invalid_splits)} rows have invalid split value: {invalid_splits['split'].unique()}")

    return errors


def generate_phase3_split(
    pairs_parquet_path: Path,
    pilot_parquet_path: Path,
    phase2a_manifest_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Generate and validate leakage-safe 4-way dataset splits (train, val, diagnostic, final_test)."""
    pairs_df = pd.read_parquet(pairs_parquet_path)
    pilot_df = pd.read_parquet(pilot_parquet_path)
    phase2a_df = pd.read_csv(phase2a_manifest_path)

    pilot_qids = set(pilot_df["source_dataset"] + "::" + pilot_df["source_id"])
    phase2a_qids = set(phase2a_df["question_id"])

    clusters = build_question_clusters(pairs_df, pilot_qids=pilot_qids)
    split_assignment = split_clusters_four_way(
        clusters,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    manifest_rows = []
    for c in clusters:
        cid = c["cluster_id"]
        sp = split_assignment[cid]
        csize = c["size"]
        for qid in c["question_ids"]:
            src, sid = qid.split("::")
            manifest_rows.append({
                "question_id": qid,
                "source_dataset": src,
                "source_id": sid,
                "cluster_id": cid,
                "cluster_size": csize,
                "split": sp,
                "in_pilot_200": qid in pilot_qids,
                "in_phase2a_cohort": qid in phase2a_qids,
            })

    manifest_df = pd.DataFrame(manifest_rows).sort_values("question_id").reset_index(drop=True)

    errors = verify_split_leakage(manifest_df, pairs_df, pilot_qids=pilot_qids, phase2a_qids=phase2a_qids)
    if errors:
        raise RuntimeError(f"Split leakage checks failed with {len(errors)} errors:\n" + "\n".join(errors))

    # Join split back to pairs table
    q_to_split = dict(zip(manifest_df["question_id"], manifest_df["split"]))
    pairs_with_split = pairs_df.copy()
    pairs_with_split["question_id"] = pairs_with_split["source_dataset"] + "::" + pairs_with_split["source_id"]
    pairs_with_split["split"] = pairs_with_split["question_id"].map(q_to_split)

    split_dfs = {
        s: pairs_with_split[pairs_with_split["split"] == s].drop(columns=["split"]).reset_index(drop=True)
        for s in SPLIT_NAMES
    }

    summary = {
        "seed": seed,
        "train_ratio_target": train_ratio,
        "val_ratio_target": val_ratio,
        "splits": SPLIT_NAMES,
        "total_questions": len(manifest_df),
        "total_clusters": len(clusters),
        "total_pairs_rows": len(pairs_df),
        "split_counts": {
            s: {
                "questions": int((manifest_df["split"] == s).sum()),
                "pct_questions": round(float((manifest_df["split"] == s).mean()), 4),
                "pairs_rows": len(split_dfs[s]),
                "by_source": manifest_df[manifest_df["split"] == s]["source_dataset"].value_counts().to_dict(),
                "pilot_questions": int((manifest_df.loc[manifest_df["split"] == s, "in_pilot_200"]).sum()),
                "phase2a_questions": int((manifest_df.loc[manifest_df["split"] == s, "in_phase2a_cohort"]).sum()),
            }
            for s in SPLIT_NAMES
        },
        "leakage_checks_passed": True,
        "n_leakage_errors": 0,
    }

    return manifest_df, split_dfs, summary
