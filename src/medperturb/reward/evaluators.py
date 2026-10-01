"""Symbolic and Task-Correctness Reward Evaluators for MedPerturb Phase 3.

Implements verifiable reward functions:
- R_correct: rewards exact task correctness under the active perturbation condition.
- R_symbolic: rewards adherence to the symbolic transformation constraints
  (semantic invariance for invariance conditions; boundary/inversion adaptation for adaptation conditions).
- R_NS: combined neuro-symbolic reward R_correct + lambda * R_symbolic.
- R_format: bonus for valid JSON output contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from medperturb.evaluation.semantic import evaluate, norm_text
from medperturb.parsing.canonical import parse
from medperturb.perturbations.registry import NOTA_TEXT, REGISTRY, output_spec_for
from medperturb.schemas.response import Example, ParseResult


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed decomposition of reward components for logging and analysis."""
    r_correct: float
    r_symbolic: float
    r_format: float
    r_total: float
    lambda_symbolic: float
    beta_format: float
    condition: str
    transition_type: str
    parse_success: bool
    is_correct: bool
    details: dict[str, Any] = field(default_factory=dict)


def compute_r_format(raw_completion: str | None) -> float:
    """Evaluates whether the completion adheres to the JSON output contract {"answer": ...}."""
    if raw_completion is None or not raw_completion.strip():
        return 0.0
    text = raw_completion.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return 1.0
    except (json.JSONDecodeError, TypeError):
        pass

    # Check for markdown codeblocks ```json ... ```
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        inner = "\n".join(lines).strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "answer" in obj:
                return 0.8  # slightly discounted for codeblock formatting
        except Exception:
            pass
    return 0.0


def compute_r_correct(example: Example, completion_text: str | None) -> tuple[float, ParseResult, Any]:
    """Computes R_correct: 1.0 if the parsed completion is strictly correct, else 0.0."""
    if example.condition == "open":
        # Open generation evaluator is unresolved; excluded from reward training
        return 0.0, ParseResult(completion_text, None, False, "open condition excluded"), None

    spec = output_spec_for(example)
    pr = parse(completion_text, spec)
    if not pr.parse_success:
        return 0.0, pr, None

    eval_res = evaluate(example, pr)
    is_corr = bool(eval_res.evaluable and eval_res.correct)
    return 1.0 if is_corr else 0.0, pr, eval_res


def compute_r_symbolic(
    example: Example,
    completion_text: str | None,
    parse_res: ParseResult | None = None,
    eval_res: Any = None,
) -> tuple[float, dict[str, Any]]:
    """Computes R_symbolic: rewards adherence to transformation semantics.

    - Invariance conditions (roman_numeral, fixed_pos, no_symbols):
      Requires interface adherence AND semantic equivalence to baseline gold option.
    - Adaptation conditions (none_of_the_provided, incorrect):
      Requires decision boundary adaptation (selecting NOTA) or task inversion (correct complement set).
    """
    cond = example.condition
    if cond == "open":
        return 0.0, {"reason": "open condition excluded from symbolic reward"}

    if parse_res is None or eval_res is None:
        spec = output_spec_for(example)
        parse_res = parse(completion_text, spec)
        if parse_res.parse_success:
            eval_res = evaluate(example, parse_res)
        else:
            eval_res = None

    if not parse_res or not parse_res.parse_success or eval_res is None:
        return 0.0, {"reason": "parse_failure", "error": getattr(parse_res, "error", "unknown")}

    trans_type = REGISTRY[cond].transition_type

    # 1. Baseline condition (mcq)
    if trans_type == "baseline":
        # Symbolic constraint is adhering to standard letter label space
        is_valid_label = parse_res.normalized_answer in example.option_labels
        score = 1.0 if is_valid_label and eval_res.correct else (0.5 if is_valid_label else 0.0)
        return score, {"valid_label": is_valid_label, "correct": eval_res.correct}

    # 2. Invariance conditions (roman_numeral, fixed_pos, no_symbols)
    if trans_type == "correctness":
        # Check interface compliance first
        interface_ok = False
        if cond == "roman_numeral":
            # Output must be a valid Roman numeral label in example.option_labels
            interface_ok = parse_res.normalized_answer in example.option_labels
        elif cond == "no_symbols":
            # Output must be valid option text
            interface_ok = eval_res.selected_option_index is not None
        elif cond == "fixed_pos":
            interface_ok = parse_res.normalized_answer in example.option_labels

        if not interface_ok:
            return 0.0, {"reason": "interface_violation", "cond": cond}

        # Semantic preservation check: does selected option text match original gold text?
        selected_text = eval_res.selected_option_text
        gold_text = example.gold_text
        semantic_match = (
            selected_text is not None
            and gold_text is not None
            and norm_text(selected_text) == norm_text(gold_text)
        )

        if semantic_match:
            return 1.0, {"interface_ok": True, "semantic_match": True}
        else:
            # Model followed format/interface but chose wrong medical answer
            return 0.25, {"interface_ok": True, "semantic_match": False}

    # 3. Adaptation: none_of_the_provided
    if cond == "none_of_the_provided":
        # Required adaptation: original correct claim was removed and replaced by NOTA.
        # Selecting NOTA proves the model adapted to the missing answer rather than picking a false distractor.
        selected_idx = eval_res.selected_option_index
        is_nota = selected_idx == example.gold_index

        if is_nota:
            return 1.0, {"adaptation_success": True, "selected_nota": True}
        else:
            # Committed required adaptation failure by picking a distractor
            return 0.0, {"adaptation_success": False, "selected_distractor": True}

    # 4. Adaptation: incorrect (task inversion)
    if cond == "incorrect":
        # Required adaptation: select ALL incorrect options (N-1 labels)
        selected_idxs = eval_res.selected_option_indices or []
        n_options = len(example.option_texts)
        expected_n = n_options - 1

        is_set_shape_ok = len(selected_idxs) == expected_n
        if not is_set_shape_ok:
            # Failed instruction inversion structure (e.g. gave single option)
            return 0.0, {"inversion_shape_ok": False, "len_selected": len(selected_idxs)}

        # Implied option: the unselected single option
        implied_idx = eval_res.implied_option_index
        if implied_idx is not None and implied_idx == example.gold_index:
            # Fully successful inversion: excluded exactly the gold option
            return 1.0, {"inversion_shape_ok": True, "implied_correct": True}
        else:
            # Followed task inversion syntax (selected N-1 options), but excluded wrong one
            return 0.5, {"inversion_shape_ok": True, "implied_correct": False}

    return 0.0, {"reason": f"unhandled condition {cond}"}


def compute_reward(
    example: Example,
    completion_text: str | None,
    lambda_symbolic: float = 1.0,
    beta_format: float = 0.0,
) -> RewardBreakdown:
    """Computes total reward and returns full component breakdown.

    R_NS = R_correct + lambda * R_symbolic + beta * R_format
    """
    r_corr, pr, eval_res = compute_r_correct(example, completion_text)
    r_symb, symb_details = compute_r_symbolic(example, completion_text, pr, eval_res)
    r_fmt = compute_r_format(completion_text) if beta_format > 0.0 else 0.0

    r_total = r_corr + (lambda_symbolic * r_symb) + (beta_format * r_fmt)

    is_correct = bool(eval_res.correct) if eval_res is not None and eval_res.evaluable else False

    details = {
        "symbolic": symb_details,
        "parse_error": getattr(pr, "error", None),
        "normalized_answer": getattr(pr, "normalized_answer", None),
    }

    return RewardBreakdown(
        r_correct=r_corr,
        r_symbolic=r_symb,
        r_format=r_fmt,
        r_total=r_total,
        lambda_symbolic=lambda_symbolic,
        beta_format=beta_format,
        condition=example.condition,
        transition_type=REGISTRY[example.condition].transition_type,
        parse_success=pr.parse_success if pr else False,
        is_correct=is_correct,
        details=details,
    )
