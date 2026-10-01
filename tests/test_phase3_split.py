"""Tests for Phase 3 leakage-safe grouped dataset splitting."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medperturb.training.split import (
    build_question_clusters,
    generate_phase3_split,
    normalize_question_text,
    split_clusters_stratified,
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
            test_ratio=0.20,
            seed=42,
            holdout_pilot=True,
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
        total_questions = sum(counts[s]["questions"] for s in ["train", "val", "test"])
        self.assertEqual(total_questions, 3154)
        self.assertEqual(counts["train"]["questions"], 2207)
        self.assertEqual(counts["val"]["questions"], 316)
        self.assertEqual(counts["test"]["questions"], 631)

    def test_zero_id_leakage(self):
        q_train = set(self.manifest_df.loc[self.manifest_df["split"] == "train", "question_id"])
        q_val = set(self.manifest_df.loc[self.manifest_df["split"] == "val", "question_id"])
        q_test = set(self.manifest_df.loc[self.manifest_df["split"] == "test", "question_id"])

        self.assertEqual(len(q_train & q_val), 0, "Train and Val share question IDs")
        self.assertEqual(len(q_train & q_test), 0, "Train and Test share question IDs")
        self.assertEqual(len(q_val & q_test), 0, "Val and Test share question IDs")

    def test_zero_normalized_text_leakage(self):
        mcq_df = self.pairs_df[["source_dataset", "source_id", "original_question"]].drop_duplicates().copy()
        mcq_df["question_id"] = mcq_df["source_dataset"] + "::" + mcq_df["source_id"]
        mcq_df["norm_q"] = mcq_df["original_question"].map(normalize_question_text)
        q_to_norm = dict(zip(mcq_df["question_id"], mcq_df["norm_q"]))

        t_train = {q_to_norm[q] for q in self.manifest_df.loc[self.manifest_df["split"] == "train", "question_id"]}
        t_val = {q_to_norm[q] for q in self.manifest_df.loc[self.manifest_df["split"] == "val", "question_id"]}
        t_test = {q_to_norm[q] for q in self.manifest_df.loc[self.manifest_df["split"] == "test", "question_id"]}

        self.assertEqual(len(t_train & t_val), 0, "Train and Val share normalized question text")
        self.assertEqual(len(t_train & t_test), 0, "Train and Test share normalized question text")
        self.assertEqual(len(t_val & t_test), 0, "Val and Test share normalized question text")

    def test_pilot_and_phase2a_held_out(self):
        pilot_qids = set(self.pilot_df["source_dataset"] + "::" + self.pilot_df["source_id"])
        phase2a_qids = set(self.phase2a_df["question_id"])

        train_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "train", "question_id"])
        val_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "val", "question_id"])
        test_qids = set(self.manifest_df.loc[self.manifest_df["split"] == "test", "question_id"])

        self.assertEqual(len(pilot_qids & train_qids), 0, "Pilot questions found in train")
        self.assertEqual(len(pilot_qids & val_qids), 0, "Pilot questions found in val")
        self.assertEqual(len(pilot_qids & test_qids), 200, "Not all pilot questions in test")

        self.assertEqual(len(phase2a_qids & train_qids), 0, "Phase 2A questions found in train")
        self.assertEqual(len(phase2a_qids & val_qids), 0, "Phase 2A questions found in val")
        self.assertEqual(len(phase2a_qids & test_qids), 50, "Not all Phase 2A questions in test")

    def test_stratification_balance(self):
        for src in ["medqa", "medmcqa", "mmlu"]:
            sub = self.manifest_df[self.manifest_df["source_dataset"] == src]
            tot = len(sub)
            tr_pct = (sub["split"] == "train").mean()
            va_pct = (sub["split"] == "val").mean()
            te_pct = (sub["split"] == "test").mean()

            self.assertAlmostEqual(tr_pct, 0.70, delta=0.015, msg=f"{src} train pct out of range: {tr_pct}")
            self.assertAlmostEqual(va_pct, 0.10, delta=0.015, msg=f"{src} val pct out of range: {va_pct}")
            self.assertAlmostEqual(te_pct, 0.20, delta=0.015, msg=f"{src} test pct out of range: {te_pct}")

    def test_pairs_row_completeness(self):
        total_rows = sum(len(df) for df in self.split_dfs.values())
        self.assertEqual(total_rows, 18924)
        self.assertEqual(len(self.split_dfs["train"]), 2207 * 6)
        self.assertEqual(len(self.split_dfs["val"]), 316 * 6)
        self.assertEqual(len(self.split_dfs["test"]), 631 * 6)

    def test_split_determinism(self):
        m2, _, _ = generate_phase3_split(
            pairs_parquet_path=PAIRS_PARQUET,
            pilot_parquet_path=PILOT_PARQUET,
            phase2a_manifest_path=PHASE2A_CSV,
            train_ratio=0.70,
            val_ratio=0.10,
            test_ratio=0.20,
            seed=42,
            holdout_pilot=True,
        )
        pd.testing.assert_frame_equal(self.manifest_df, m2)


if __name__ == "__main__":
    unittest.main()
