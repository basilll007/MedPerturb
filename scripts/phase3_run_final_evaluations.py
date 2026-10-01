"""Phase 3 Master Evaluation and Synthesis Runner.

Executes untouched final_test evaluations across:
1. Base Qwen3-4B
2. Policy A: Correctness-trained Qwen3-4B
3. Policy B: Neuro-Symbolic Qwen3-4B

Then computes:
- Paired statistical tests (McNemar, bootstrap CIs)
- Quantitative and qualitative error analysis
- Manuscript figures and tables
- Dashboard summary updates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medperturb.evaluation.paired_stats import compute_paired_comparison_report
from scripts.phase3_analysis_and_figures import (
    generate_adaptation_failure_figure,
    generate_invariance_adaptation_tradeoff_figure,
    generate_latex_and_csv_tables,
    generate_per_perturbation_figure,
    generate_pipeline_figure,
)
from scripts.phase3_error_analysis import perform_error_analysis
from scripts.phase3_eval_policy import evaluate_policy

CHECKPOINTS_DIR = ROOT / "results" / "phase3" / "checkpoints"
EVALS_DIR = ROOT / "results" / "phase3" / "evaluations"
STATS_DIR = ROOT / "results" / "phase3" / "statistics"
FIGURES_DIR = ROOT / "figures" / "phase3"
TABLES_DIR = ROOT / "tables" / "phase3"


def run_complete_final_evaluation(stage: str = "final_test") -> None:
    print(f"============================================================")
    print(f"PHASE E: FINAL EVALUATION ON UNTOUCHED '{stage.upper()}' COHORT")
    print(f"============================================================\n")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_corr = CHECKPOINTS_DIR / "policy_correctness"
    ckpt_ns = CHECKPOINTS_DIR / "policy_neurosymbolic"

    if not ckpt_corr.exists():
        raise FileNotFoundError(f"Missing Policy A checkpoint at {ckpt_corr}")
    if not ckpt_ns.exists():
        raise FileNotFoundError(f"Missing Policy B checkpoint at {ckpt_ns}")

    # 1. Evaluate Base Model
    print("\n--- 1/3: Evaluating Base Qwen3-4B ---")
    base_dir = evaluate_policy(policy_name="base", stage=stage, adapter_path=None)

    # 2. Evaluate Policy A (Correctness)
    print("\n--- 2/3: Evaluating Policy A (Correctness-Trained) ---")
    corr_dir = evaluate_policy(policy_name="correctness", stage=stage, adapter_path=ckpt_corr)

    # 3. Evaluate Policy B (Neuro-Symbolic)
    print("\n--- 3/3: Evaluating Policy B (Neuro-Symbolic) ---")
    ns_dir = evaluate_policy(policy_name="neurosymbolic", stage=stage, adapter_path=ckpt_ns)

    # Load paired and response data
    base_resp = pd.read_csv(base_dir / "responses.csv")
    base_paired = pd.read_csv(base_dir / "paired_results.csv")
    base_metrics = pd.read_csv(base_dir / "metrics_by_perturbation.csv")

    corr_resp = pd.read_csv(corr_dir / "responses.csv")
    corr_paired = pd.read_csv(corr_dir / "paired_results.csv")
    corr_metrics = pd.read_csv(corr_dir / "metrics_by_perturbation.csv")

    ns_resp = pd.read_csv(ns_dir / "responses.csv")
    ns_paired = pd.read_csv(ns_dir / "paired_results.csv")
    ns_metrics = pd.read_csv(ns_dir / "metrics_by_perturbation.csv")

    # 4. Statistical Hypothesis Testing
    print("\n============================================================")
    print("PHASE E: COMPUTING PAIRED STATISTICAL HYPOTHESIS TESTS")
    print("============================================================\n")

    # Primary Comparison: Correctness vs Neuro-Symbolic
    comp_ns_vs_corr = compute_paired_comparison_report(
        resp_a=corr_resp,
        resp_b=ns_resp,
        paired_a=corr_paired,
        paired_b=ns_paired,
        model_a_label="Correctness",
        model_b_label="Neuro-Symbolic",
    )

    # Secondary Comparison: Base vs Neuro-Symbolic
    comp_ns_vs_base = compute_paired_comparison_report(
        resp_a=base_resp,
        resp_b=ns_resp,
        paired_a=base_paired,
        paired_b=ns_paired,
        model_a_label="Base",
        model_b_label="Neuro-Symbolic",
    )

    # Baseline Comparison: Base vs Correctness
    comp_corr_vs_base = compute_paired_comparison_report(
        resp_a=base_resp,
        resp_b=corr_resp,
        paired_a=base_paired,
        paired_b=corr_paired,
        model_a_label="Base",
        model_b_label="Correctness",
    )

    stats_report = {
        "stage": stage,
        "primary_hypothesis_test_NS_vs_Correctness": comp_ns_vs_corr,
        "NS_vs_Base": comp_ns_vs_base,
        "Correctness_vs_Base": comp_corr_vs_base,
    }

    with open(STATS_DIR / f"paired_hypothesis_test_{stage}.json", "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2)

    print(f"[+] Saved hypothesis test report to {STATS_DIR}")

    # 5. Error Analysis (Phase F)
    print("\n============================================================")
    print("PHASE F: STRUCTURED ERROR AND FAILURE ANALYSIS")
    print("============================================================\n")
    error_analysis_dir = ROOT / "results" / "phase3" / "error_analysis" / f"stage_{stage}"
    perform_error_analysis(
        base_dir=base_dir,
        correctness_dir=corr_dir,
        neurosymbolic_dir=ns_dir,
        out_dir=error_analysis_dir,
    )

    # 6. Figures and Tables Generation (Phase G)
    print("\n============================================================")
    print("PHASE G: GENERATING PUBLICATION FIGURES AND TABLES")
    print("============================================================\n")
    models_paired = {
        "Base": base_paired,
        "Correctness": corr_paired,
        "Neuro-Symbolic": ns_paired,
    }
    models_metrics = {
        "Base": base_metrics,
        "Correctness": corr_metrics,
        "Neuro-Symbolic": ns_metrics,
    }
    models_resp = {
        "Base": base_resp,
        "Correctness": corr_resp,
        "Neuro-Symbolic": ns_resp,
    }

    generate_pipeline_figure(FIGURES_DIR / "fig1_pipeline")
    generate_adaptation_failure_figure(models_paired, FIGURES_DIR / f"fig2_adaptation_failure_{stage}")
    generate_per_perturbation_figure(models_metrics, FIGURES_DIR / f"fig3_per_perturbation_{stage}")
    generate_invariance_adaptation_tradeoff_figure(models_paired, FIGURES_DIR / f"fig4_tradeoff_{stage}")
    generate_latex_and_csv_tables(models_resp, models_paired, models_metrics, TABLES_DIR)

    # Print Primary Hypothesis Result Summary
    pri = comp_ns_vs_corr["primary_outcome_adaptation_failure"]
    print("\n============================================================")
    print("PRIMARY HYPOTHESIS TEST RESULTS (Neuro-Symbolic vs Correctness)")
    print("============================================================")
    print(f"Correctness Failure Rate: {pri['rate_a'] * 100.0:.2f}%")
    print(f"Neuro-Symbolic Failure Rate: {pri['rate_b'] * 100.0:.2f}%")
    print(f"Absolute Delta: {pri['delta_rate'] * 100.0:.2f}%")
    print(f"Relative Reduction: {pri['relative_reduction_pct']:.2f}%")
    print(f"McNemar Test p-value: {pri['mcnemar_success_test']['p_value']}")
    print(f"Bootstrap 95% CI: [{pri['bootstrap_delta_failure']['ci_lower']:.4f}, {pri['bootstrap_delta_failure']['ci_upper']:.4f}]")
    print(f"Statistically Significant (p < 0.05): {pri['mcnemar_success_test']['significant_at_05']}")
    print("============================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Master Final Evaluation.")
    parser.add_argument("--stage", type=str, default="final_test", choices=["final_test", "diagnostic", "val"])
    args = parser.parse_args()

    run_complete_final_evaluation(stage=args.stage)


if __name__ == "__main__":
    main()
