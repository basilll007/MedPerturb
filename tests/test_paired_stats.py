"""Unit tests for paired statistical testing (McNemar and Bootstrap CIs)."""

import numpy as np
import pytest
from medperturb.evaluation.paired_stats import mcnemar_test, paired_bootstrap_ci


def test_mcnemar_identical_outcomes():
    a = [True, False, True, True, False]
    b = [True, False, True, True, False]
    res = mcnemar_test(a, b)
    assert res["discordant"] == 0
    assert res["p_value"] == 1.0
    assert not res["significant_at_05"]


def test_mcnemar_known_discordant():
    # 20 pairs: A only correct in 1, B only correct in 10
    a = [True] * 1 + [False] * 10 + [True] * 5 + [False] * 4
    b = [False] * 1 + [True] * 10 + [True] * 5 + [False] * 4
    res = mcnemar_test(a, b)
    assert res["discordant"] == 11
    assert res["a_only"] == 1
    assert res["b_only"] == 10
    assert res["p_value"] < 0.05
    assert res["significant_at_05"]


def test_paired_bootstrap_ci():
    a = np.array([0.2, 0.3, 0.2, 0.4, 0.1, 0.2, 0.3, 0.1] * 10)
    b = np.array([0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9] * 10)
    ci_res = paired_bootstrap_ci(a, b, n_bootstraps=500, seed=42)
    assert ci_res["observed_delta"] > 0.5
    assert ci_res["ci_lower"] > 0.4
    assert ci_res["ci_upper"] > 0.5
    assert ci_res["excludes_zero"] is True
