"""Parser registry: one parser per answer shape. Parsers only extract and
normalize; they never score. Ambiguity is always a parse failure, never a guess.
"""

import json
import re

from medperturb.schemas.response import OutputSpec, ParseResult

PARSER_VERSION = "phase2a_parsers_v1"


def _json_answer(raw: str):
    """Returns (found, value). Our output contract is {"answer": ...}."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, None
    if isinstance(obj, dict) and "answer" in obj:
        return True, obj["answer"]
    return False, None


def _label_text_fallback(text: str, labels: list[str]) -> ParseResult:
    """Byte-for-byte the validated Phase 1 logic (scripts/phase1a_parse.py)."""
    text = text.strip()
    stripped = re.sub(r"^[\s(\[]*", "", text)
    stripped = re.sub(r"[\s)\].,:;!]*$", "", stripped)
    if stripped.upper() in labels:
        return ParseResult(text, stripped.upper(), True, None)
    ordered = sorted(labels, key=len, reverse=True)
    matches = re.findall(r"\b(" + "|".join(ordered) + r")\b", text.upper())
    if not matches:
        return ParseResult(text, None, False, "no valid label found")
    if len(set(matches)) == 1:
        return ParseResult(text, matches[0], True, None)
    return ParseResult(text, None, False, f"ambiguous: multiple labels {sorted(set(matches))}")


def parse_label(raw: str | None, spec: OutputSpec) -> ParseResult:
    if raw is None or not raw.strip():
        return ParseResult(raw, None, False, "empty response")
    labels = list(spec.allowed_labels)
    found, value = _json_answer(raw)
    if found:
        if not isinstance(value, str):
            return ParseResult(value, None, False, f"answer is {type(value).__name__}, expected a single label")
        return _label_text_fallback(value, labels)
    return _label_text_fallback(raw, labels)


def parse_label_set(raw: str | None, spec: OutputSpec) -> ParseResult:
    if raw is None or not raw.strip():
        return ParseResult(raw, None, False, "empty response")
    labels = list(spec.allowed_labels)
    found, value = _json_answer(raw)
    if found and isinstance(value, list):
        items = [str(v).strip().upper() for v in value]
    elif found and isinstance(value, str):
        items = re.findall(r"\b(" + "|".join(sorted(labels, key=len, reverse=True)) + r")\b", value.upper())
    else:
        return ParseResult(raw, None, False, "response not valid JSON answer object")
    if not items:
        return ParseResult(value, None, False, "empty label set")
    invalid = [x for x in items if x not in labels]
    if invalid:
        return ParseResult(value, None, False, f"invalid labels {invalid}")
    return ParseResult(value, sorted(set(items), key=labels.index), True, None)


def parse_free_text(raw: str | None, spec: OutputSpec) -> ParseResult:
    if raw is None or not raw.strip():
        return ParseResult(raw, None, False, "empty response")
    found, value = _json_answer(raw)
    if not found:
        return ParseResult(raw, None, False, "response not valid JSON answer object")
    if not isinstance(value, str) or not value.strip():
        return ParseResult(value, None, False, "answer empty or not a string")
    return ParseResult(value, value.strip(), True, None)


PARSERS = {"label": parse_label, "label_set": parse_label_set, "free_text": parse_free_text}


def parse(raw: str | None, spec: OutputSpec) -> ParseResult:
    return PARSERS[spec.kind](raw, spec)
