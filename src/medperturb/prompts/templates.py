"""Answer-only prompt templates, one per display/output combination.

No chain-of-thought, no request for reasoning. Labeled single-answer
conditions reproduce the validated Phase 1 template byte-for-byte
(scripts/phase1a_prompts.py). ReMedQA's own prompt/prompt_think fields are
never reused: they request step-by-step reasoning.
"""

import hashlib

from medperturb.perturbations.registry import REGISTRY
from medperturb.schemas.response import Example

PROMPT_VERSION = "phase2a_v1"

_TAIL = "Output only that single token and nothing else - no explanation, no punctuation, no reasoning."


def _label_list(labels):
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def build_prompt(ex: Example) -> str:
    spec = REGISTRY[ex.condition]
    lines = [f"Question: {ex.question}", ""]

    if spec.display == "labeled":
        for lab, txt in zip(ex.option_labels, ex.option_texts):
            lines.append(f"({lab}) {txt}")
        lines.append("")
        kind = "roman numeral" if spec.label_style == "roman" else "letter"
        if spec.output_kind == "label":
            lines.append(f"Answer with only the {kind} of the correct option ({_label_list(ex.option_labels)}). {_TAIL}")
        else:  # label_set: task inversion
            n_wrong = len(ex.option_labels) - 1
            lines.append(
                f"Exactly {n_wrong} of these options are INCORRECT. Answer with only the {kind}s of all the "
                f"incorrect options. Output only those {kind}s and nothing else - no explanation, no reasoning.")
    elif spec.display == "bullets":
        for txt in ex.option_texts:
            lines.append(f"- {txt}")
        lines.append("")
        lines.append("Answer with only the exact text of the correct option, copied verbatim, without any letter "
                     "or symbol. Output only that text and nothing else - no explanation, no reasoning.")
    else:  # open: no options shown
        lines.append("Answer with only the concise final answer (a short phrase). Output only that answer and "
                     "nothing else - no explanation, no reasoning.")
    return "\n".join(lines)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
