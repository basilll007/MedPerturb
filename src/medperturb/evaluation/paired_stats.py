"""Paired statistical analysis module for MedPerturb Phase 3.

Provides hypothesis testing for paired experimental designs:
- Exact McNemar test (with continuity correction / binomial exact test)
- Paired bootstrap confidence intervals (BCa / percentile)
- Comprehensive comparative report between model cohorts
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def mcnemar_test(
    outcomes_a: list[bool] | np.ndarray,
    outcomes_b: list[bool] | np.ndarray,
) -> dict[str, Any]:
    """Computes paired McNemar test between Model A and Model B on identical questions.

    Null Hypothesis (H0): Model A and Model B have identical marginal probabilities of success.
    """
    a = np.asarray(outcomes_a, dtype=bool)
    b = np.asarray(outcomes_b, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match for paired test: len(a)={len(a)}, len(b)={len(b)}")

    n = len(a)
    n_both = int(np.sum(a & b))
    n_neither = int(np.sum(~a & ~b))
    n_a_only = int(np.sum(a & ~b))  # A correct, B incorrect (b in standard 2x2 notation)
    n_b_only = int(np.sum(~a & b))  # A incorrect, B correct (c in standard 2x2 notation)

    discordant = n_a_only + n_b_only

    if discordant == 0:
        return {
            "n_total": n,
            "both_correct": n_both,
            "both_incorrect": n_neither,
            "a_only": n_a_only,
            "b_only": n_b_only,
            "discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "test_type": "exact_binomial",
            "significant_at_05": False,
            "significant_at_01": False,
        }

    # Use exact binomial test if discordant count < 25, else Edwards' continuity-corrected chi2
    if discordant < 25:
        binom_res = stats.binomtest(n_b_only, discordant, p=0.5, alternative="two-sided")
        p_val = float(binom_res.pvalue)
        stat = float(n_b_only)
        test_type = "exact_binomial"
    else:
        # Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
        chi2_stat = float(((abs(n_a_only - n_b_only) - 1.0) ** 2) / discordant)
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))
        stat = chi2_stat
        test_type = "mcnemar_continuity_corrected"

    return {
        "n_total": n,
        "both_correct": n_both,
        "both_incorrect": n_neither,
        "a_only": n_a_only,
        "b_only": n_b_only,
        "discordant": discordant,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "test_type": test_type,
        "significant_at_05": bool(p_val < 0.05),
        "significant_at_01": bool(p_val < 0.01),
    }


def paired_bootstrap_ci(
    outcomes_a: list[float] | np.ndarray,
    outcomes_b: list[float] | np.ndarray,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes paired bootstrap confidence interval for Delta = Mean(B) - Mean(A)."""
    a = np.asarray(outcomes_a, dtype=float)
    b = np.asarray(outcomes_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Lengths must match for paired bootstrap")

    n = len(a)
    diff = b - a
    observed_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        idxs = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diff[idxs])

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_diffs, lower_pct))
    ci_upper = float(np.percentile(boot_diffs, upper_pct))
    std_err = float(np.std(boot_diffs))

    # Two-sided empirical p-value for difference != 0
    p_empirical = 2.0 * min(
        float(np.mean(boot_diffs <= 0.0)),
        float(np.mean(boot_diffs >= 0.0)),
    )
    p_empirical = min(1.0, max(1.0 / n_bootstraps, p_empirical))

    excludes_zero = bool((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))

    return {
        "observed_mean_a": round(float(np.mean(a)), 4),
        "observed_mean_b": round(float(np.mean(b)), 4),
        "observed_delta": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std_error": round(std_err, 4),
        "confidence_level": ci,
        "n_bootstraps": n_bootstraps,
        "p_empirical": round(p_empirical, 5),
        "excludes_zero": excludes_zero,
    }


def compute_paired_comparison_report(
    resp_a: pd.DataFrame,
    resp_b: pd.DataFrame,
    paired_a: pd.DataFrame,
    paired_b: pd.DataFrame,
    model_a_label: str = "Model_A",
    model_b_label: str = "Model_B",
) -> dict[str, Any]:
    """Generates a complete paired statistical report between Model A and Model B.

    Evaluates:
    - Primary outcome: required_adaptation_failure reduction
    - Invariance accuracy preservation
    - Per-condition accuracy comparisons
    """
    # 1. Primary Outcome: Required Adaptation Failure
    # Defined on adaptation conditions: none_of_the_provided and incorrect
    adapt_conds = ["none_of_the_provided", "incorrect"]

    # Filter to adaptation conditions
    p_a_adapt = paired_a[paired_a["condition"].isin(adapt_conds)].copy()
    p_b_adapt = paired_b[paired_b["condition"].isin(adapt_conds)].copy()

    # Align on (question_id, condition)
    merged_adapt = pd.merge(
        p_a_adapt,
        p_b_adapt,
        on=["question_id", "condition"],
        suffixes=("_a", "_b"),
    )

    # In paired_results: adaptation strings starting with "failure" indicate required adaptation failure
    fail_a = merged_adapt["adaptation_a"].fillna("").str.startswith("failure").to_numpy(dtype=bool)
    fail_b = merged_adapt["adaptation_b"].fillna("").str.startswith("failure").to_numpy(dtype=bool)

    # Adaptation success is (~fail)
    succ_a = (merged_adapt["adaptation_a"] == "successful_adaptation").to_numpy(dtype=bool)
    succ_b = (merged_adapt["adaptation_b"] == "successful_adaptation").to_numpy(dtype=bool)

    primary_mcnemar = mcnemar_test(succ_a, succ_b)
    primary_boot = paired_bootstrap_ci(
        fail_a.astype(float),
        fail_b.astype(float),
    )

    # 2. Invariance Preservation: roman_numeral, fixed_pos, no_symbols
    invar_conds = ["roman_numeral", "fixed_pos", "no_symbols"]
    merged_invar = pd.merge(
        paired_a[paired_a["condition"].isin(invar_conds)],
        paired_b[paired_b["condition"].isin(invar_conds)],
        on=["question_id", "condition"],
        suffixes=("_a", "_b"),
    )

    corr_invar_a = merged_invar["perturbed_correct_a"].fillna(False).to_numpy(dtype=bool)
    corr_invar_b = merged_invar["perturbed_correct_b"].fillna(False).to_numpy(dtype=bool)

    invar_mcnemar = mcnemar_test(corr_invar_a, corr_invar_b)
    invar_boot = paired_bootstrap_ci(
        corr_invar_a.astype(float),
        corr_invar_b.astype(float),
    )

    # 3. Baseline MCQ Performance
    merged_mcq = pd.merge(
        resp_a[resp_a["condition"] == "mcq"],
        resp_b[resp_b["condition"] == "mcq"],
        on="question_id",
        suffixes=("_a", "_b"),
    )
    mcq_a = merged_mcq["correct_a"].fillna(False).to_numpy(dtype=bool)
    mcq_b = merged_mcq["correct_b"].fillna(False).to_numpy(dtype=bool)

    mcq_mcnemar = mcnemar_test(mcq_a, mcq_b)
    mcq_boot = paired_bootstrap_ci(
        mcq_a.astype(float),
        mcq_b.astype(float),
    )

    # 4. Per-Perturbation Condition Comparisons
    all_conds = sorted(list(set(resp_a["condition"].unique()).intersection(resp_b["condition"].unique())))
    per_condition = {}
    for c in all_conds:
        sub_a = resp_a[resp_a["condition"] == c]
        sub_b = resp_b[resp_b["condition"] == c]
        merged_c = pd.merge(sub_a, sub_b, on="question_id", suffixes=("_a", "_b"))
        c_a = merged_c["correct_a"].fillna(False).to_numpy(dtype=bool)
        c_b = merged_c["correct_b"].fillna(False).to_numpy(dtype=bool)
        per_condition[c] = {
            "n": len(merged_c),
            "acc_a": round(float(np.mean(c_a)), 4),
            "acc_b": round(float(np.mean(c_b)), 4),
            "delta": round(float(np.mean(c_b) - np.mean(c_a)), 4),
            "mcnemar": mcnemar_test(c_a, c_b),
            "bootstrap": paired_bootstrap_ci(c_a.astype(float), c_b.astype(float)),
        }

    return {
        "model_a": model_a_label,
        "model_b": model_b_label,
        "primary_outcome_adaptation_failure": {
            "rate_a": round(float(np.mean(fail_a)), 4),
            "rate_b": round(float(np.mean(fail_b)), 4),
            "delta_rate": round(float(np.mean(fail_b) - np.mean(fail_a)), 4),
            "relative_reduction_pct": round(
                float((np.mean(fail_a) - np.mean(fail_b)) / (np.mean(fail_a) + 1e-9) * 100.0), 2
            ),
            "mcnemar_success_test": primary_mcnemar,
            "bootstrap_delta_failure": primary_boot,
        },
        "secondary_outcome_invariance": {
            "acc_a": round(float(np.mean(corr_invar_a)), 4),
            "acc_b": round(float(np.mean(corr_invar_b)), 4),
            "delta_acc": round(float(np.mean(corr_invar_b) - np.mean(corr_invar_a)), 4),
            "mcnemar": invar_mcnemar,
            "bootstrap": invar_boot,
        },
        "secondary_outcome_mcq": {
            "acc_a": round(float(np.mean(mcq_a)), 4),
            "acc_b": round(float(np.mean(mcq_b)), 4),
            "delta_acc": round(float(np.mean(mcq_b) - np.mean(mcq_a)), 4),
            "mcnemar": mcq_mcnemar,
            "bootstrap": mcq_boot,
        },
        "per_condition": per_condition,
    }
