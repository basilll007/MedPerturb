"""Pre-run semantic audit: verify each condition's observed transformation
against the registry's claimed taxonomy, using both the example contents and
ReMedQA's own dataset prompts. Returns per-(question, condition) rows plus a
list of contradictions; any contradiction must stop execution."""

import json
import re

from medperturb.perturbations.registry import REGISTRY, ROMAN, NOTA_TEXT, CONDITIONS


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _observed_display(prompt: str) -> str:
    if re.search(r"\((A|I)\)\s", prompt):
        return "labeled"
    if re.search(r"\n- \S", prompt):
        return "bullets"
    return "none"


def _observed_format(raw_gold) -> str:
    s = str(raw_gold).strip()
    if s.startswith("[") and s.endswith("]"):
        return "label_set"
    if s in ROMAN:
        return "roman_label"
    if re.fullmatch(r"[A-J]", s):
        return "letter_label"
    return "text"


EXPECTED_FORMAT = {
    "mcq": "letter_label", "roman_numeral": "roman_label", "none_of_the_provided": "letter_label",
    "fixed_pos": "letter_label", "incorrect": "label_set", "open": "text", "no_symbols": "text",
}


def audit_question(examples: dict, pilot_rows) -> tuple[list[dict], list[str]]:
    rows_by_pt = {r["perturbation_type"]: r for _, r in pilot_rows.iterrows()}
    mcq = examples["mcq"]
    n = len(mcq.option_texts)
    out, contradictions = [], []

    for cond in CONDITIONS:
        ex = examples[cond]
        spec = REGISTRY[cond]
        if cond == "mcq":
            md = json.loads(rows_by_pt["roman_numeral"]["metadata"])
            prompt = md.get("original_prompt_think", "") + "\n" + md.get("original_prompt", "")
            raw_gold = mcq.gold_label
        else:
            md = json.loads(rows_by_pt[cond]["metadata"])
            prompt = md.get("perturbed_prompt_think", "") + "\n" + md.get("perturbed_prompt", "")
            raw_gold = rows_by_pt[cond]["perturbed_answer_raw"]

        problems = []
        q_changed = _norm(ex.question) != _norm(mcq.question)
        same_order = ex.option_texts == mcq.option_texts
        same_set = sorted(ex.option_texts) == sorted(mcq.option_texts)
        gold_text = ex.gold_text
        gold_content_changed = None
        gold_position_changed = None

        obs_display = _observed_display(prompt)
        obs_format = _observed_format(raw_gold)
        obs_task_inverted = bool(re.search(r"\bincorrect\b", prompt, re.I))

        if obs_display != spec.display:
            problems.append(f"display observed={obs_display} expected={spec.display}")
        if obs_format != EXPECTED_FORMAT[cond]:
            problems.append(f"answer format observed={obs_format} expected={EXPECTED_FORMAT[cond]}")
        if obs_task_inverted != (cond == "incorrect"):
            problems.append(f"task inversion observed={obs_task_inverted}")

        if cond == "mcq":
            if mcq.option_labels != [chr(65 + i) for i in range(n)]:
                problems.append("mcq labels not consecutive letters")
            gold_content_changed, gold_position_changed = False, False
        elif cond == "roman_numeral":
            if q_changed: problems.append("question changed")
            if not same_order: problems.append("option texts/order changed")
            if ex.option_labels != ROMAN[:n]: problems.append(f"labels {ex.option_labels}")
            gold_content_changed = gold_text != mcq.gold_text
            gold_position_changed = ex.gold_index != mcq.gold_index
            if gold_content_changed or gold_position_changed: problems.append("gold moved or changed")
        elif cond == "none_of_the_provided":
            if q_changed: problems.append("question changed")
            if len(ex.option_texts) != n: problems.append("option count changed")
            gold_content_changed = gold_text != mcq.gold_text
            gold_position_changed = ex.gold_index != mcq.gold_index
            if gold_position_changed: problems.append("gold position changed")
            if gold_text != NOTA_TEXT: problems.append(f"gold text is {gold_text!r}")
            if mcq.gold_text in ex.option_texts: problems.append("original gold text still present")
            others_same = all(ex.option_texts[i] == mcq.option_texts[i] for i in range(n) if i != mcq.gold_index)
            if not others_same: problems.append("non-gold options changed")
        elif cond == "fixed_pos":
            if q_changed: problems.append("question changed")
            if not same_set: problems.append("option set changed")
            gold_content_changed = gold_text != mcq.gold_text
            gold_position_changed = ex.gold_index != mcq.gold_index
            if gold_content_changed: problems.append("gold content changed")
            if ex.gold_index != n - 1: problems.append(f"gold not in last position (index {ex.gold_index})")
        elif cond == "incorrect":
            if q_changed: problems.append("question changed")
            if not same_order: problems.append("option texts/order changed")
            expected = [i for i in range(n) if i != mcq.gold_index]
            if ex.gold_indices != expected: problems.append(f"gold set {ex.gold_label} != complement of mcq gold")
            gold_content_changed, gold_position_changed = False, False
        elif cond == "open":
            gold_content_changed = gold_text != mcq.gold_text
            gold_position_changed = None  # no displayed options
            if gold_content_changed: problems.append("gold text differs from mcq gold text")
        elif cond == "no_symbols":
            if q_changed: problems.append("question changed")
            if not same_order: problems.append("option texts/order changed")
            gold_content_changed = gold_text != mcq.gold_text
            gold_position_changed = ex.gold_index != mcq.gold_index
            if gold_content_changed or gold_position_changed: problems.append("gold moved or changed")

        dup_texts = len(set(_norm(t) for t in ex.option_texts)) < len(ex.option_texts)
        out.append({
            "question_id": ex.question_id, "source_dataset": ex.source_dataset, "condition": cond,
            "category": spec.category, "question_text_changed": q_changed,
            "option_count": len(ex.option_texts), "option_labels": "|".join(ex.option_labels),
            "option_order_same_as_mcq": same_order, "option_set_same_as_mcq": same_set,
            "gold_label": json.dumps(ex.gold_label), "gold_text": gold_text, "gold_index": ex.gold_index,
            "gold_content_changed": gold_content_changed, "gold_position_changed": gold_position_changed,
            "task_objective_changed": obs_task_inverted,
            "observed_display": obs_display, "observed_response_format": obs_format,
            "duplicate_option_texts": dup_texts,
            "expectation_met": not problems, "contradictions": "; ".join(problems),
        })
        contradictions += [f"{ex.question_id} {cond}: {p}" for p in problems]
    return out, contradictions
