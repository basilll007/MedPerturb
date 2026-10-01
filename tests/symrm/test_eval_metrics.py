"""Unit tests for tie-band scoring metrics, CFR, and headline DFR computation."""

import pytest
from symrm.eval.metrics import (
    compute_pairwise_accuracy,
    compute_counterfactual_flip_metrics,
    compute_discriminative_flip_metrics,
    get_default_epsilon,
)


def test_epsilon_loaded():
    eps = get_default_epsilon()
    assert 0.10 <= eps <= 0.15, f"Epsilon {eps} should be approx 0.10 - 0.15"


def test_pairwise_accuracy_with_tie_band():
    eps = 0.125
    chosen = [1.0, 0.5, 0.55, 0.50]
    rejected = [0.5, 1.0, 0.50, 0.55]

    res = compute_pairwise_accuracy(chosen, rejected, epsilon=eps)
    assert res.num_pairs == 4
    assert res.num_correct == 1
    assert res.num_incorrect == 1
    assert res.num_ties == 2
    assert res.strict_accuracy == 0.25
    assert res.tie_rate == 0.50


def test_counterfactual_flip_metrics():
    eps = 0.125
    base_def = [1.0, 1.0, 0.55, 0.5]
    base_alt = [0.5, 0.5, 0.50, 1.0]

    edited_def = [0.5, 1.0, 0.5, 1.0]
    edited_alt = [1.0, 0.5, 0.5, 0.5]

    res = compute_counterfactual_flip_metrics(base_def, base_alt, edited_def, edited_alt, epsilon=eps)
    assert res.num_counterfactual_pairs == 4
    assert res.num_flips == 1
    assert res.num_default_dominant == 1
    assert res.num_ties == 1
    assert res.num_both_incorrect == 1
    assert res.cfr == 0.25
    assert res.default_dominance_rate == 0.25


def test_discriminative_flip_metrics():
    eps = 0.125
    # 4 scenarios:
    # 1. Gold standard DFR: base correct (+0.5), edited correct (+0.5), near-miss correct (+0.5) -> DFR=1, Over-flip=0
    # 2. Over-flip (keyword reactor): base correct (+0.5), edited correct (+0.5), near-miss WRONG (-0.5) -> DFR=0, Over-flip=1, CFR=1
    # 3. Default dominant: base correct (+0.5), edited WRONG (-0.5), near-miss correct (+0.5) -> DFR=0, Over-flip=0, CFR=0
    # 4. Tie on near-miss: base correct (+0.5), edited correct (+0.5), near-miss tie (+0.05) -> DFR=0, Over-flip=0, CFR=1
    base_def = [1.0, 1.0, 1.0, 1.0]
    base_alt = [0.5, 0.5, 0.5, 0.5]

    edited_def = [0.5, 0.5, 1.0, 0.5]
    edited_alt = [1.0, 1.0, 0.5, 1.0]

    nm_def = [1.0, 0.5, 1.0, 0.55]
    nm_alt = [0.5, 1.0, 0.5, 0.50]

    res = compute_discriminative_flip_metrics(
        base_def, base_alt, edited_def, edited_alt, nm_def, nm_alt, epsilon=eps
    )
    assert res.num_scenarios == 4
    assert res.num_dfr_correct == 1
    assert res.num_over_flips == 1
    assert res.num_cfr_flips == 3  # scenarios 1, 2, 4 have base & edited correct
    assert res.dfr == 0.25
    assert res.over_flip_rate == 0.25
    assert res.cfr == 0.75
