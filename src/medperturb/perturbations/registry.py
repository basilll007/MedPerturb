"""Explicit perturbation registry.

Every condition declares what it changes and which metrics are semantically
justified for it. Nothing downstream is allowed to assume that two
conditions are interchangeable robustness tests.
"""

import ast
import json
from dataclasses import dataclass

from medperturb.schemas.response import Example, OutputSpec


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    category: str
    description: str
    display: str               # "labeled" | "bullets" | "none"
    label_style: str | None    # "letter" | "roman" | None
    output_kind: str           # "label" | "label_set" | "free_text"
    evaluator: str             # "single_option" | "option_set_complement" | "option_text" | "open_text"
    transition_type: str       # "baseline" | "correctness" | "adaptation" | "inversion" | "open_generation"
    option_semantics_preserved: bool   # same option texts available as in mcq
    gold_content_preserved: bool       # gold answer text identical to mcq gold text
    recon_defined: bool                # "same semantic answer as mcq" is a meaningful expectation
    notes: str = ""


REGISTRY: dict[str, PerturbationSpec] = {
    "mcq": PerturbationSpec(
        "mcq", "baseline", "Original multiple-choice representation, letter labels.",
        "labeled", "letter", "label", "single_option", "baseline", True, True, True),
    "roman_numeral": PerturbationSpec(
        "roman_numeral", "presentation / answer-label transformation",
        "Option labels A-D become I-IV; question, option texts and order unchanged.",
        "labeled", "roman", "label", "single_option", "correctness", True, True, True),
    "none_of_the_provided": PerturbationSpec(
        "none_of_the_provided", "answer-set / decision-boundary transformation",
        "Gold option's text replaced by 'None of the provided options'; position unchanged.",
        "labeled", "letter", "label", "single_option", "adaptation", False, False, False,
        notes="Same index != same semantic answer: the original correct content no longer exists."),
    "fixed_pos": PerturbationSpec(
        "fixed_pos", "answer-position diagnostic",
        "Options reordered so the gold answer sits in the last position (D).",
        "labeled", "letter", "label", "single_option", "correctness", True, True, True,
        notes="Deterministic gold-at-D placement; semantic identity compared by option text, not index."),
    "incorrect": PerturbationSpec(
        "incorrect", "task/instruction inversion",
        "Model must name ALL incorrect options instead of the correct one.",
        "labeled", "letter", "label_set", "option_set_complement", "inversion", True, True, False,
        notes="Correct iff selected set == all non-gold options; implied answer = the unselected option."),
    "open": PerturbationSpec(
        "open", "response-space / MCQ-to-generation transformation",
        "Reworded question with no options shown; free-text answer.",
        "none", None, "free_text", "open_text", "open_generation", True, True, False,
        notes="Deterministic string evaluator is provisional; unmatched answers are evaluation-ambiguous, not wrong."),
    "no_symbols": PerturbationSpec(
        "no_symbols", "output-interface / answer-format transformation",
        "Options shown as unlabeled bullets; answer must be the option text.",
        "bullets", None, "free_text", "option_text", "correctness", True, True, True),
}

CONDITIONS = list(REGISTRY.keys())
ROMAN = ["I", "II", "III", "IV", "V", "VI"]
NOTA_TEXT = "None of the provided options"


def output_spec_for(example: Example) -> OutputSpec:
    spec = REGISTRY[example.condition]
    if spec.output_kind in ("label", "label_set"):
        return OutputSpec(spec.output_kind, tuple(example.option_labels))
    return OutputSpec("free_text", ())


def _options(json_str: str) -> dict:
    return json.loads(json_str)


def build_examples(pilot_rows_for_question) -> dict[str, Example]:
    """Build all 7 Examples for one underlying question from its pilot rows
    (one row per non-mcq perturbation; mcq comes from the shared original_*
    columns, which must be identical across rows)."""
    rows = {r["perturbation_type"]: r for _, r in pilot_rows_for_question.iterrows()}
    missing = [c for c in CONDITIONS if c != "mcq" and c not in rows]
    if missing:
        raise ValueError(f"missing perturbation rows: {missing}")

    any_row = rows["roman_numeral"]
    for c, r in rows.items():
        if (r["original_question"], r["original_options"], r["original_answer_letter"]) != (
                any_row["original_question"], any_row["original_options"], any_row["original_answer_letter"]):
            raise ValueError(f"mcq baseline differs between perturbation rows ({c})")

    sd, sid = any_row["source_dataset"], any_row["source_id"]
    qid = f"{sd}::{sid}"
    out = {}

    mcq_opts = _options(any_row["original_options"])
    labels, texts = list(mcq_opts.keys()), list(mcq_opts.values())
    g = any_row["original_answer_letter"]
    out["mcq"] = Example(qid, sd, sid, "mcq", any_row["original_question"], labels, texts, True,
                         g, mcq_opts[g], labels.index(g))

    for cond in ("roman_numeral", "none_of_the_provided", "fixed_pos"):
        r = rows[cond]
        o = _options(r["perturbed_options"])
        labels, texts = list(o.keys()), list(o.values())
        g = r["perturbed_answer_raw"]
        out[cond] = Example(qid, sd, sid, cond, r["perturbed_question"], labels, texts, True,
                            g, o.get(g), labels.index(g) if g in labels else None)

    r = rows["incorrect"]
    o = _options(r["perturbed_options"])
    labels, texts = list(o.keys()), list(o.values())
    gold_set = sorted(ast.literal_eval(r["perturbed_answer_raw"]))
    out["incorrect"] = Example(qid, sd, sid, "incorrect", r["perturbed_question"], labels, texts, True,
                               gold_set, None, None, [labels.index(x) for x in gold_set])

    for cond, displayed in (("open", False), ("no_symbols", True)):
        r = rows[cond]
        o = _options(r["perturbed_options"])
        texts = list(o.values())
        gold_text = r["perturbed_answer_raw"]
        out[cond] = Example(qid, sd, sid, cond, r["perturbed_question"], [], texts, displayed,
                            None, gold_text, texts.index(gold_text) if gold_text in texts else None)
    return out
