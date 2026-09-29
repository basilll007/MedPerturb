"""Phase 1A prompt templates.

Deliberately NOT reusing ReMedQA's own `prompt`/`prompt_think` fields, which
request step-by-step chain-of-thought reasoning ("Solve them in a step-by-step
fashion...", "Enclose your final answer within \\boxed{}"). Phase 1A requires
no-CoT, deterministic, single-token answers.
"""

import json

PROMPT_VERSION = "phase1a_v1"

_LABEL_SETS = {
    "mcq": ["A", "B", "C", "D"],
    "roman_numeral": ["I", "II", "III", "IV"],
}


def build_prompt(perturbation_type: str, question: str, options: dict, labels: list[str] | None = None) -> str:
    """`labels` overrides the static lookup - required for perturbations like
    none_of_the_provided where the valid label set must be derived from the
    actual example's options rather than assumed (Phase 1B requirement:
    "Derive valid labels from the actual perturbed example")."""
    if labels is None:
        if perturbation_type not in _LABEL_SETS:
            raise ValueError(f"No Phase 1A prompt template defined for perturbation_type={perturbation_type!r} "
                              f"and no explicit labels were provided")
        labels = _LABEL_SETS[perturbation_type]

    lines = [f"Question: {question}", ""]
    for label in labels:
        lines.append(f"({label}) {options[label]}")
    lines.append("")
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    label_kind = "roman numeral" if perturbation_type == "roman_numeral" else "letter"
    lines.append(
        f"Answer with only the {label_kind} "
        f"of the correct option ({label_list}). "
        "Output only that single token and nothing else - no explanation, no punctuation, no reasoning."
    )
    return "\n".join(lines)


def parse_options_json(options_str: str) -> dict:
    return json.loads(options_str)


def get_labels(perturbation_type: str) -> list[str]:
    return list(_LABEL_SETS[perturbation_type])


def labels_from_options(options: dict) -> list[str]:
    """Derive valid labels directly from an example's own options dict,
    for perturbations (e.g. none_of_the_provided) not in the static
    _LABEL_SETS - never assume a fixed A-D/I-IV set for those."""
    return list(options.keys())
