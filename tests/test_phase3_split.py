"""Tests for Phase 3 4-way leakage-safe grouped dataset splitting."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medperturb.training.split import (
    SPLIT_NAMES,
    build_question_clusters,
    generate_phase3_split,
    normalize_question_text,
    split_clusters_four_way,
    verify_split_leakage,
)

PAIRS_PARQUET = ROOT / "data" / "processed" / "remedqa_pairs.parquet"
PILOT_PARQUET = ROOT / "data" / "processed" / "pilot_200.parquet"
PHASE2A_CSV = ROOT / "data" / "processed" / "phase2a_50_manifest.csv"


class TestPhase3Split(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert PAIRS_PARQUET.exists(), f"Missing {PAIRS_PARQUET}"
        assert PILOT_PARQUET.exists(), f"Missing {PILOT_PARQUET}"
        assert PHASE2A_CSV.exists(), f"Missing {PHASE2A_CSV}"

        cls.pairs_df = pd.read_parquet(PAIRS_PARQUET)
        cls.pilot_df = pd.read_parquet(PILOT_PARQUET)
        cls.phase2a_df = pd.read_csv(PHASE2A_CSV)

        cls.manifest_df, cls.split_dfs, cls.summary = generate_phase3_split(
            pairs_parquet_path=PAIRS_PARQUET,
            pilot_parquet_path=PILOT_PARQUET,
            phase2a_manifest_path=PHASE2A_CSV,
            train_ratio=0.70,
            val_ratio=0.10,
            seed=42,
        )

    def test_normalization(self):
        self.assertEqual(normalize_question_text("  What is X? "), "what is x")
        self.assertEqual(normalize_question_text("Hello, World! (123)"), "hello world 123")
        self.assertEqual(normalize_question_text("A/B/C: test\nline"), "a b c test line")

    def test_cluster_connected_components(self):
        clusters = build_question_clusters(self.pairs_df)
        total_q = sum(c["size"] for c in clusters)
        self.assertEqual(total_q, 3154)
        self.assertEqual(len(clusters), 3084)
        for c in clusters:
            sources = {q.split("::")[0] for q in c["question_ids"]}
            self.assertEqual(len(sources), 1, f"Cluster crosses sources: {sources}")

    def test_question_counts(self):
        counts = self.summary["split_counts"]
        total_questions = sum(counts[s]["questions"] for s in SPLIT_NAMES)
        self.assertEqual(total_questions, 3154)
        self.assertEqual(counts["train"]["questions"], 2207)
        self.assertEqual(counts["val"]["questions"], 317)
        self.assertEqual(counts["diagnostic"]["questions"], 207)
        self.assertEqual(counts["final_test"]["questions"], 423)

    def test_zero_id_leakage(self):
        qids = {s: set(self.manifest_df.loc[self.manifest_df["split"] == s, "question_id"]) for s in SPLIT_NAMES}
        for i, s1 in enumerate(SPLIT_NAMES):
            for s2 in SPLIT_NAMES[i + 1:]:
                intersection = qids[s1] & qids[s2]
                self.assertEqual(len(intersection), 0, f"Overlapping IDs between {s1} and {s2}: {intersection}")

    def test_zero_normalized_text_leakage(self):
        mcq_df = self.pairs_df[["source_dataset", "source_id", "original_question"]].drop_duplicates().copy()
        mcq_df["question_id"] = mcq_df["source_dataset"] + "::" + mcq_df["source_id"]
        mcq_df["norm_q"] = mcq_df["original_question"].map(normalize_question_text)
        q_to_norm = dict(zip(mcq_df["question_id"], mcq_df["norm_q"]))

        norms = {
            s: {q_to_norm[q] for q in self.manifest_df.loc[self.manifest_df["split"] == s, "question_id"]}
            for s in SPLIT_NAMES
        }
        for i, s1 in enumerate(SPLIT_NAMES):
            for s2 in SPLIT_NAMES[i + 1:]:
                intersection = norms[s1] & norms[s2]
                self.assertEqual(len(intersection), 0, f"Shared stems between {s1} and {s2}: {intersection}")

    def test_pilot_and_phase2a_isolation_in_diagnostic(self):
        pilot_qids = set(self.pilot_df["source_dataset"] + "::" + self.pilot_df["source_id"])
        phase2a_qids = set(self.phase2a_df["question_id"])

        diag_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "diagnostic", "question_id"])
        train_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "train", "question_id"])
        val_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "val", "question_id"])
        final_test_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "final_test", "question_id"])

        # Must be in diagnostic
        self.assertEqual(len(pilot_qids & diag_qids), 200, "Pilot questions missing from diagnostic")
        self.assertEqual(len(phase2a_qids & diag_qids), 50, "Phase 2A questions missing from diagnostic")

        # Must NOT be in train, val, or final_test
        self.assertEqual(len(pilot_qids & train_qids), 0, "Pilot in train")
        self.assertEqual(len(pilot_qids & val_qids), 0, "Pilot in val")
        self.assertEqual(len(pilot_qids & final_test_qids), 0, "Pilot in final_test")

        self.assertEqual(len(phase2a_qids & train_qids), 0, "Phase 2A in train")
        self.assertEqual(len(phase2a_qids & val_qids), 0, "Phase 2A in val")
        self.assertEqual(len(phase2a_qids & final_test_qids), 0, "Phase 2A in final_test")

    def test_stratification_balance(self):
        for src in ["medqa", "medmcqa", "mmlu"]:
            sub = self.manifest_df[self.manifest_df["source_dataset"] == src]
            tr_pct = (sub["split"] == "train").mean()
            va_pct = (sub["split"] == "val").mean()

            self.assertAlmostEqual(tr_pct, 0.70, delta=0.015, msg=f"{src} train pct out of range: {tr_pct}")
            self.assertAlmostEqual(va_pct, 0.10, delta=0.015, msg=f"{src} val pct out of range: {va_pct}")

    def test_pairs_row_completeness(self):
        total_rows = sum(len(df) for df in self.split_dfs.values())
        self.assertEqual(total_rows, 18924)
        self.assertEqual(len(self.split_dfs["train"]), 2207 * 6)
        self.assertEqual(len(self.split_dfs["val"]), 317 * 6)
        self.assertEqual(len(self.split_dfs["diagnostic"]), 207 * 6)
        self.assertEqual(len(self.split_dfs["final_test"]), 423 * 6)

    def test_split_determinism(self):
        m2, _, _ = generate_phase3_split(
            pairs_parquet_path=PAIRS_PARQUET,
            pilot_parquet_path=PILOT_PARQUET,
            phase2a_manifest_path=PHASE2A_CSV,
            train_ratio=0.70,
            val_ratio=0.10,
            seed=42,
        )
        pd.testing.assert_frame_equal(self.manifest_df, m2)


if __name__ == "__main__":
    unittest.main()
