"""Semantic evaluators: map a parsed answer to option identity and correctness.

Tracks position identity (selected_option_index) AND semantic identity
(selected_option_text) separately - they coincide only when a condition
preserves option semantics.
"""

import re

from medperturb.perturbations.registry import REGISTRY
from medperturb.schemas.response import EvalResult, Example, ParseResult

EVALUATOR_VERSION = "phase2a_evaluators_v1"
MIN_COVER_OPTION_IN_ANSWER = 0.4
MIN_COVER_ANSWER_IN_OPTION = 0.6


def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _contains_words(hay: str, needle: str) -> bool:
    return bool(needle) and re.search(r"(?:^| )" + re.escape(needle) + r"(?: |$)", hay) is not None


def match_text(answer: str, option_texts: list[str]):
    """Returns (index | None, method). Deliberately conservative: nested
    options (e.g. MMLU 'X' vs 'X and Y') resolve to the maximal match only."""
    a = norm_text(answer)
    opts = [norm_text(t) for t in option_texts]
    exact = [i for i, o in enumerate(opts) if o == a]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return None, "ambiguous_exact"

    cands = [i for i, o in enumerate(opts) if _contains_words(a, o)]
    maximal = [i for i in cands if not any(j != i and _contains_words(opts[j], opts[i]) for j in cands)]
    if len(maximal) == 1 and len(opts[maximal[0]]) >= MIN_COVER_OPTION_IN_ANSWER * len(a):
        return maximal[0], "option_in_answer"
    if len(maximal) > 1:
        return None, "ambiguous_multiple_options_in_answer"

    rev = [i for i, o in enumerate(opts) if _contains_words(o, a)]
    if len(rev) == 1 and len(a) >= MIN_COVER_ANSWER_IN_OPTION * len(opts[rev[0]]):
        return rev[0], "answer_in_option"
    return None, "no_match"


def evaluate(ex: Example, pr: ParseResult) -> EvalResult:
    spec = REGISTRY[ex.condition]
    if not pr.parse_success:
        return EvalResult(False, None, None, None, None, None, None, "not parsed")

    if spec.evaluator == "single_option":
        idx = ex.option_labels.index(pr.normalized_answer)
        return EvalResult(True, idx == ex.gold_index, idx, ex.option_texts[idx], None, None, "label")

    if spec.evaluator == "option_set_complement":
        idxs = [ex.option_labels.index(x) for x in pr.normalized_answer]
        n = len(ex.option_texts)
        implied = next(iter(set(range(n)) - set(idxs))) if len(idxs) == n - 1 else None
        return EvalResult(True, sorted(idxs) == sorted(ex.gold_indices), None,
                          ex.option_texts[implied] if implied is not None else None, idxs, implied, "label_set",
                          None if len(idxs) == n - 1 else f"selected {len(idxs)} options, expected {n - 1}")

    if spec.evaluator == "option_text":  # options shown: an unmatched answer is an invalid option, i.e. wrong
        idx, method = match_text(pr.normalized_answer, ex.option_texts)
        if idx is None:
            return EvalResult(True, False, None, None, None, None, method, "answer matches no displayed option")
        return EvalResult(True, idx == ex.gold_index, idx, ex.option_texts[idx], None, None, method)

    if spec.evaluator == "open_text":  # options hidden: unmatched is evaluation ambiguity, not wrong
        idx, method = match_text(pr.normalized_answer, ex.option_texts)
        if idx is None:
            return EvalResult(False, None, None, None, None, None, method,
                              "free-text answer matches no reference option; needs semantic judgment")
        return EvalResult(True, idx == ex.gold_index, idx, ex.option_texts[idx], None, None, method)

    raise ValueError(f"unknown evaluator {spec.evaluator}")
