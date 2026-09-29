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


def build_prompt(perturbation_type: str, question: str, options: dict) -> str:
    if perturbation_type not in _LABEL_SETS:
        raise ValueError(f"No Phase 1A prompt template defined for perturbation_type={perturbation_type!r}")

    labels = _LABEL_SETS[perturbation_type]
    lines = [f"Question: {question}", ""]
    for label in labels:
        lines.append(f"({label}) {options[label]}")
    lines.append("")
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    lines.append(
        f"Answer with only the {'letter' if perturbation_type == 'mcq' else 'roman numeral'} "
        f"of the correct option ({label_list}). "
        "Output only that single token and nothing else - no explanation, no punctuation, no reasoning."
    )
    return "\n".join(lines)


def parse_options_json(options_str: str) -> dict:
    return json.loads(options_str)


def get_labels(perturbation_type: str) -> list[str]:
    return list(_LABEL_SETS[perturbation_type])
