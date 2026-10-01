"""Scoring and evaluation metrics with tie-band tolerance for SymRM.

Per user requirements:
- epsilon = 2 * max observed batched/unbatched score diff from padding_invariance_eval.json (approx 0.13).
- Any |margin| < epsilon counts as a tie.
- Headline metric: Discriminative Flip Rate (DFR) = fraction of scenarios where base,
  edited, AND near-miss are all decisively correct (> epsilon in correct direction).
- Over-flip Rate = edited correct (> epsilon) but near-miss wrong (< -epsilon).
- Secondary metric: Counterfactual Flip Rate (CFR) = base and edited both correct (> epsilon).
- Pairwise accuracy with tie-band tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel


def get_default_epsilon(eval_path: str = "results/symrm/padding_invariance_eval.json") -> float:
    """Computes epsilon = 2 * max_abs_score_diff from padding_invariance_eval.json."""
    p = Path(eval_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        max_diff = data.get("max_abs_score_diff", 0.0625)
        return float(2.0 * max_diff)
    return 0.1250


class ScoringResult(BaseModel):
    num_pairs: int
    epsilon: float
    num_correct: int
    num_incorrect: int
    num_ties: int
    strict_accuracy: float  # correct / total
    tie_rate: float         # ties / total
    mean_margin: float


class CounterfactualPairResult(BaseModel):
    num_counterfactual_pairs: int
    epsilon: float
    num_flips: int          # M_base > +eps and M_edited > +eps
    num_default_dominant: int  # M_base > +eps and M_edited < -eps
    num_alternative_dominant: int # M_base < -eps and M_edited > +eps
    num_both_incorrect: int # M_base < -eps and M_edited < -eps
    num_ties: int           # either |M_base| <= eps or |M_edited| <= eps
    cfr: float              # Counterfactual Flip Rate
    default_dominance_rate: float
    tie_rate: float
    mean_base_margin: float
    mean_edited_margin: float


class DiscriminativeFlipResult(BaseModel):
    num_scenarios: int
    epsilon: float
    num_dfr_correct: int       # base, edited, and near-miss all decisively correct (> eps)
    num_over_flips: int        # edited correct (> eps) but near-miss wrong (< -eps)
    num_cfr_flips: int         # base and edited correct (> eps)
    num_any_ties: int
    dfr: float                 # Discriminative Flip Rate = num_dfr_correct / num_scenarios
    over_flip_rate: float      # Over-flip Rate = num_over_flips / num_scenarios
    cfr: float                 # Counterfactual Flip Rate = num_cfr_flips / num_scenarios
    mean_base_margin: float
    mean_edited_margin: float
    mean_near_miss_margin: float


def compute_pairwise_accuracy(
    chosen_scores: List[float],
    rejected_scores: List[float],
    epsilon: Optional[float] = None
) -> ScoringResult:
    """Computes pairwise accuracy with tie band [-epsilon, +epsilon]."""
    if epsilon is None:
        epsilon = get_default_epsilon()

    assert len(chosen_scores) == len(rejected_scores), "Score arrays must match in length"
    n = len(chosen_scores)
    if n == 0:
        return ScoringResult(
            num_pairs=0, epsilon=epsilon, num_correct=0, num_incorrect=0, num_ties=0,
            strict_accuracy=0.0, tie_rate=0.0, mean_margin=0.0
        )

    margins = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    correct = sum(1 for m in margins if m > epsilon)
    incorrect = sum(1 for m in margins if m < -epsilon)
    ties = sum(1 for m in margins if abs(m) <= epsilon)

    return ScoringResult(
        num_pairs=n,
        epsilon=epsilon,
        num_correct=correct,
        num_incorrect=incorrect,
        num_ties=ties,
        strict_accuracy=float(correct / n),
        tie_rate=float(ties / n),
        mean_margin=float(sum(margins) / n),
    )


def compute_counterfactual_flip_metrics(
    base_default_scores: List[float],
    base_alternative_scores: List[float],
    edited_default_scores: List[float],
    edited_alternative_scores: List[float],
    epsilon: Optional[float] = None
) -> CounterfactualPairResult:
    """Computes Counterfactual Flip Rate (CFR) and Default Dominance with tie band."""
    if epsilon is None:
        epsilon = get_default_epsilon()

    n = len(base_default_scores)
    assert len(base_alternative_scores) == n
    assert len(edited_default_scores) == n
    assert len(edited_alternative_scores) == n

    if n == 0:
        return CounterfactualPairResult(
            num_counterfactual_pairs=0, epsilon=epsilon, num_flips=0, num_default_dominant=0,
            num_alternative_dominant=0, num_both_incorrect=0, num_ties=0, cfr=0.0,
            default_dominance_rate=0.0, tie_rate=0.0, mean_base_margin=0.0, mean_edited_margin=0.0
        )

    # Base: default is chosen, alternative is rejected
    base_margins = [d - a for d, a in zip(base_default_scores, base_alternative_scores)]
    # Edited: alternative is chosen, default is rejected
    edited_margins = [a - d for a, d in zip(edited_alternative_scores, edited_default_scores)]

    flips = 0
    default_dom = 0
    alt_dom = 0
    both_inc = 0
    ties = 0

    for m_b, m_e in zip(base_margins, edited_margins):
        if abs(m_b) <= epsilon or abs(m_e) <= epsilon:
            ties += 1
        elif m_b > epsilon and m_e > epsilon:
            flips += 1
        elif m_b > epsilon and m_e < -epsilon:
            default_dom += 1
        elif m_b < -epsilon and m_e > epsilon:
            alt_dom += 1
        elif m_b < -epsilon and m_e < -epsilon:
            both_inc += 1

    return CounterfactualPairResult(
        num_counterfactual_pairs=n,
        epsilon=epsilon,
        num_flips=flips,
        num_default_dominant=default_dom,
        num_alternative_dominant=alt_dom,
        num_both_incorrect=both_inc,
        num_ties=ties,
        cfr=float(flips / n),
        default_dominance_rate=float(default_dom / n),
        tie_rate=float(ties / n),
        mean_base_margin=float(sum(base_margins) / n),
        mean_edited_margin=float(sum(edited_margins) / n),
    )


def compute_discriminative_flip_metrics(
    base_default_scores: List[float],
    base_alternative_scores: List[float],
    edited_default_scores: List[float],
    edited_alternative_scores: List[float],
    near_miss_default_scores: List[float],
    near_miss_alternative_scores: List[float],
    epsilon: Optional[float] = None
) -> DiscriminativeFlipResult:
    """Computes headline Discriminative Flip Rate (DFR) and Over-flip Rate across triplet scenarios.

    For scenario s:
    - Base margin: M_base = r(default) - r(alternative). Correct if M_base > +eps.
    - Edited margin: M_edited = r(alternative) - r(default). Correct if M_edited > +eps.
    - Near-miss margin: M_nm = r(default) - r(alternative). Correct if M_nm > +eps.

    DFR = fraction where base, edited, AND near-miss are all decisively correct.
    Over-flip Rate = fraction where edited is correct (> eps) BUT near-miss is wrong (< -eps).
    """
    if epsilon is None:
        epsilon = get_default_epsilon()

    n = len(base_default_scores)
    assert len(base_alternative_scores) == n
    assert len(edited_default_scores) == n
    assert len(edited_alternative_scores) == n
    assert len(near_miss_default_scores) == n
    assert len(near_miss_alternative_scores) == n

    if n == 0:
        return DiscriminativeFlipResult(
            num_scenarios=0, epsilon=epsilon, num_dfr_correct=0, num_over_flips=0,
            num_cfr_flips=0, num_any_ties=0, dfr=0.0, over_flip_rate=0.0, cfr=0.0,
            mean_base_margin=0.0, mean_edited_margin=0.0, mean_near_miss_margin=0.0
        )

    base_margins = [d - a for d, a in zip(base_default_scores, base_alternative_scores)]
    edited_margins = [a - d for a, d in zip(edited_alternative_scores, edited_default_scores)]
    near_miss_margins = [d - a for d, a in zip(near_miss_default_scores, near_miss_alternative_scores)]

    dfr_count = 0
    over_flip_count = 0
    cfr_count = 0
    tie_count = 0

    for m_b, m_e, m_nm in zip(base_margins, edited_margins, near_miss_margins):
        b_correct = m_b > epsilon
        e_correct = m_e > epsilon
        nm_correct = m_nm > epsilon

        b_tie = abs(m_b) <= epsilon
        e_tie = abs(m_e) <= epsilon
        nm_tie = abs(m_nm) <= epsilon

        if b_tie or e_tie or nm_tie:
            tie_count += 1

        # CFR: base and edited both correct
        if b_correct and e_correct:
            cfr_count += 1

        # DFR: base, edited, and near-miss all correct
        if b_correct and e_correct and nm_correct:
            dfr_count += 1

        # Over-flip: edited correct, but near-miss wrong (reacts to keyword rather than logic)
        if e_correct and (m_nm < -epsilon):
            over_flip_count += 1

    return DiscriminativeFlipResult(
        num_scenarios=n,
        epsilon=epsilon,
        num_dfr_correct=dfr_count,
        num_over_flips=over_flip_count,
        num_cfr_flips=cfr_count,
        num_any_ties=tie_count,
        dfr=float(dfr_count / n),
        over_flip_rate=float(over_flip_count / n),
        cfr=float(cfr_count / n),
        mean_base_margin=float(sum(base_margins) / n),
        mean_edited_margin=float(sum(edited_margins) / n),
        mean_near_miss_margin=float(sum(near_miss_margins) / n),
    )
