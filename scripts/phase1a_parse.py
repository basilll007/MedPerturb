"""Phase 1A answer parsing - deliberately separate from inference.

Never silently treats an unparseable response as wrong; returns an explicit
parsing_status so the caller can log/exclude it correctly.
"""

import re
from dataclasses import dataclass

_LABEL_SETS = {
    "mcq": ["A", "B", "C", "D"],
    "roman_numeral": ["I", "II", "III", "IV"],
}


@dataclass
class ParseResult:
    parsed_label: str | None
    parsing_status: str  # "ok" | "unparseable" | "ambiguous"


def parse_answer(raw_text: str, perturbation_type: str, labels: list[str] | None = None) -> ParseResult:
    """`labels` overrides the static lookup - required for perturbations
    (e.g. none_of_the_provided) whose valid label set must be derived from
    the actual example rather than assumed. When omitted, behavior is
    byte-identical to the validated Phase 1A parser."""
    if raw_text is None:
        return ParseResult(None, "unparseable")

    if labels is None:
        labels = _LABEL_SETS[perturbation_type]
    text = raw_text.strip()

    # Fast path: response is exactly the label (optionally wrapped in
    # parens/punctuation), which is what we asked for.
    stripped = re.sub(r"^[\s(\[]*", "", text)
    stripped = re.sub(r"[\s)\].,:;!]*$", "", stripped)
    if stripped.upper() in labels:
        return ParseResult(stripped.upper(), "ok")

    # Fallback: search for standalone label tokens anywhere in the text.
    # Longer roman numerals must be checked before shorter ones (IV/III
    # before I) to avoid a substring false-match.
    ordered = sorted(labels, key=len, reverse=True)
    pattern = r"\b(" + "|".join(ordered) + r")\b"
    matches = re.findall(pattern, text.upper())

    if not matches:
        return ParseResult(None, "unparseable")

    distinct = set(matches)
    if len(distinct) == 1:
        return ParseResult(matches[0], "ok")

    return ParseResult(None, "ambiguous")
