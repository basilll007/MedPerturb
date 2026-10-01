"""Unit test verifying the keyword heuristic baseline behavior.

Asserts:
1. On edited items containing trigger keywords, the heuristic predicts alternative (1) for 100% of items.
2. On base items, the heuristic predicts default (0) for 100% of items.
3. On near-miss items, the heuristic predicts alternative (1) >= 70% of the time,
   confirming that near-miss items successfully bait the keyword shortcut.
"""

import json
from pathlib import Path
import pytest
from symrm.eval.lexical_shortcuts import keyword_heuristic_predict, TRIGGER_KEYWORDS


def test_keyword_heuristic_on_edited_items():
    data_path = Path("results/symrm/data/all_items.jsonl")
    assert data_path.exists()

    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # On edited items containing a trigger keyword, the heuristic must predict alternative (1)
    edited_with_keywords = [
        it for it in items
        if it["item_type"] == "edited" and any(kw in it["prompt"].lower() for kw in TRIGGER_KEYWORDS)
    ]
    # Pregnancy (90) + Allergy (90) = 180 items with trigger keywords
    assert len(edited_with_keywords) == 180

    failures = []
    for it in edited_with_keywords:
        pred = keyword_heuristic_predict(it["prompt"])
        if pred != 1:
            failures.append(it["item_id"])

    assert len(failures) == 0, f"Keyword heuristic failed to predict alternative on {len(failures)} edited items: {failures[:5]}"


def test_keyword_heuristic_on_base_items():
    data_path = Path("results/symrm/data/all_items.jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    base_items = [it for it in items if it["item_type"] == "base"]
    assert len(base_items) == 300

    base_correct = sum(1 for it in base_items if keyword_heuristic_predict(it["prompt"]) == 0)
    assert base_correct == 300, f"Expected 300 base items predicted default (0), got {base_correct}"


def test_keyword_heuristic_on_near_miss_items():
    data_path = Path("results/symrm/data/all_items.jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    nm_with_keywords = [
        it for it in items
        if it["item_type"] == "near_miss" and any(kw in it["prompt"].lower() for kw in TRIGGER_KEYWORDS)
    ]
    assert len(nm_with_keywords) == 180

    # Near-miss items containing the trigger keyword 100% over-flip to the alternative action
    predicted_alt = sum(1 for it in nm_with_keywords if keyword_heuristic_predict(it["prompt"]) == 1)
    rate = predicted_alt / len(nm_with_keywords)
    assert rate == 1.0, f"Expected 100% over-flip on keyword-bearing near-miss items, got {rate*100:.1f}%"

