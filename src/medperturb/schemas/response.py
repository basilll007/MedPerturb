"""Provider-neutral data contracts shared by every framework component."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutputSpec:
    """What shape of answer a condition expects. Adapters translate this into
    whatever constrained-output mechanism their provider supports."""
    kind: str  # "label" | "label_set" | "free_text"
    allowed_labels: tuple[str, ...] = ()


@dataclass
class Example:
    question_id: str
    source_dataset: str
    source_id: str
    condition: str
    question: str
    option_labels: list[str]        # labels in display order; [] when not labeled
    option_texts: list[str]         # option texts in display order (kept for scoring even if not displayed)
    options_displayed: bool
    gold_label: Any                 # str | list[str] | None
    gold_text: str | None
    gold_index: int | None          # index into option_texts
    gold_indices: list[int] | None = None  # label-set conditions (incorrect)


@dataclass
class ModelResponse:
    provider: str
    model: str
    model_version: str | None
    question_id: str
    condition: str
    raw_response: str | None
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    latency_s: float | None
    timestamp: str
    cost_usd: float | None
    raw_payload: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class ParseResult:
    raw_answer: Any
    normalized_answer: Any
    parse_success: bool
    parse_failure_reason: str | None


@dataclass
class EvalResult:
    evaluable: bool                 # False => evaluation ambiguity / not scoreable
    correct: bool | None
    selected_option_index: int | None
    selected_option_text: str | None
    selected_option_indices: list[int] | None
    implied_option_index: int | None  # incorrect: the single option NOT marked incorrect
    match_method: str | None
    note: str | None = None
