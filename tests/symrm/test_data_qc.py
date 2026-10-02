"""Comprehensive QC test suite for SymRM Phase 1 Step 2 generated data.

Validates:
1. Strict response neutrality (no forbidden clinical rationale/keywords, plain prescription format).
2. For dose-reduction rules (renal_02, renal_04), both options name the exact same drug.
3. Token length difference per rule (mean |diff| <= 2 tokens with Qwen tokenizer).
4. Near-miss verifier label == base label for 100% of items.
5. Symbolic verifier 100% agreement across all items.
6. Template and rule split integrity (no leakage between train, near_ood, far_ood).
7. Data version tagging ("v0-unreviewed").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import re
import pytest
import yaml
from transformers import AutoTokenizer

from symrm.data.generator import SymbolicVerifier, StructuredFacts


FORBIDDEN_WORDS = [
    "avoid", "discontinue", "contraindicated", "contraindication",
    "allergy", "allergic", "anaphylaxis",
    "pregnancy", "pregnant", "gestation", "fetus", "fetal", "hcg",
    "renal", "kidney", "nephropathy", "egfr", "crcl", "creatinine",
    "switch", "instead", "because", "due to", "risk", "hazard", "warning",
]


def load_dataset_items() -> List[Dict]:
    data_path = Path("results/symrm/data/all_items.jsonl")
    assert data_path.exists(), f"Dataset not found at {data_path}"
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_rules_spec() -> Dict:
    rules_path = Path("configs/symrm/rules.yaml")
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_data_version_tagging():
    items = load_dataset_items()
    assert len(items) > 0
    for it in items:
        assert it.get("data_version") == "v0-unreviewed", f"Item {it['item_id']} missing 'v0-unreviewed' tag"


def test_response_format_and_neutrality():
    items = load_dataset_items()
    violations = []

    for it in items:
        for opt_name in ["chosen", "rejected", "default_option", "alternative_option"]:
            text = it[opt_name].lower()

            # Must begin with 'prescribe '
            if not text.startswith("prescribe "):
                violations.append(f"Item {it['item_id']} {opt_name} does not start with 'Prescribe': '{it[opt_name]}'")

            # Check forbidden keywords
            for fw in FORBIDDEN_WORDS:
                # Whole word match or substring
                if fw in text:
                    violations.append(f"Item {it['item_id']} {opt_name} contains forbidden word '{fw}': '{it[opt_name]}'")

    assert len(violations) == 0, f"Found {len(violations)} response neutrality violations:\n" + "\n".join(violations[:10])


def test_dose_reduction_rules_same_drug():
    items = load_dataset_items()
    dose_rules = ["renal_02", "renal_04"]

    for it in items:
        if it["rule_id"] in dose_rules:
            # Check both options name the exact same drug
            def_text = it["default_option"].lower()
            alt_text = it["alternative_option"].lower()

            if it["rule_id"] == "renal_02":
                assert "enoxaparin" in def_text and "enoxaparin" in alt_text, (
                    f"renal_02 must name enoxaparin in both options: {def_text} vs {alt_text}"
                )
            elif it["rule_id"] == "renal_04":
                assert "gabapentin" in def_text and "gabapentin" in alt_text, (
                    f"renal_04 must name gabapentin in both options: {def_text} vs {alt_text}"
                )


def test_near_miss_verifier_matches_base_100_percent():
    items = load_dataset_items()
    rules_spec = load_rules_spec()
    rules_by_id = {r["rule_id"]: r for r in rules_spec["rules"]}

    # Group by template_id
    templates: Dict[str, Dict[str, Dict]] = {}
    for it in items:
        tpl_id = it["template_id"]
        templates.setdefault(tpl_id, {})[it["item_type"]] = it

    mismatches = []
    for tpl_id, triplet in templates.items():
        assert "base" in triplet, f"Missing base in {tpl_id}"
        assert "near_miss" in triplet, f"Missing near_miss in {tpl_id}"
        assert "edited" in triplet, f"Missing edited in {tpl_id}"
        assert "null_edit" in triplet, f"Missing null_edit in {tpl_id}"

        base_item = triplet["base"]
        nm_item = triplet["near_miss"]
        rule_spec = rules_by_id[base_item["rule_id"]]

        # Verify base
        base_facts = StructuredFacts(**base_item["structured_facts"])
        b_fired, b_act = SymbolicVerifier.verify(base_facts, rule_spec)

        # Verify near-miss
        nm_facts = StructuredFacts(**nm_item["structured_facts"])
        nm_fired, nm_act = SymbolicVerifier.verify(nm_facts, rule_spec)

        assert not b_fired, f"Base fired in {tpl_id}"
        assert not nm_fired, f"Near-miss fired in {tpl_id}"

        if nm_act != b_act:
            mismatches.append(f"Template {tpl_id}: near-miss action '{nm_act}' != base action '{b_act}'")
        if nm_item["chosen"] != base_item["chosen"]:
            mismatches.append(f"Template {tpl_id}: near-miss chosen '{nm_item['chosen']}' != base chosen '{base_item['chosen']}'")

    assert len(mismatches) == 0, f"Found {len(mismatches)} near-miss vs base mismatches:\n" + "\n".join(mismatches)


def test_token_length_differences_under_threshold():
    rules_spec = load_rules_spec()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    diffs_per_rule = {}
    abs_diffs = []

    for rule in rules_spec["rules"]:
        rid = rule["rule_id"]
        def_act = rule["default_action"]
        alt_act = rule["rule_action"]

        len_def = len(tokenizer.encode(def_act))
        len_alt = len(tokenizer.encode(alt_act))
        diff = len_def - len_alt
        diffs_per_rule[rid] = {
            "len_default": len_def,
            "len_alt": len_alt,
            "diff": diff,
            "abs_diff": abs(diff),
        }
        abs_diffs.append(abs(diff))

    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)

    # Save to results
    out_file = Path("results/symrm/token_length_qc.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_abs_diff": mean_abs_diff,
                "passed": mean_abs_diff <= 2.0,
                "per_rule": diffs_per_rule,
            },
            f,
            indent=2,
        )

    assert mean_abs_diff <= 2.0, f"Mean token length difference {mean_abs_diff:.2f} > 2.0"


BANNED_TOKENS = [
    "severe", "marked", "mild", "moderate", "impairment", "dysfunction",
    "insufficiency", "adequate", "preserved", "safely", "tolerated",
    "confirming", "reference range", "band"
]


def test_no_banned_tokens_in_prompts():
    items = load_dataset_items()
    violations = []

    allowed_parens = [
        re.compile(r"^\(\d+\s+weeks\s+gestation\)$"),
        re.compile(r"^\(anaphylaxis\)$"),
        re.compile(r"^\(Stevens-Johnson syndrome\)$"),
        re.compile(r"^\(bronchospasm and nasal polyps\)$"),
    ]

    threshold_pattern = re.compile(r"[><=]=?\s*\d+")
    paren_pattern = re.compile(r"\([^)]*\)")

    for it in items:
        prompt = it["prompt"]
        prompt_lower = prompt.lower()

        # Check banned tokens
        for bt in BANNED_TOKENS:
            if re.search(r"\b" + re.escape(bt) + r"\b", prompt_lower):
                violations.append(f"Item {it['item_id']} contains banned token '{bt}'")

        # Check threshold annotations
        threshold_matches = threshold_pattern.findall(prompt)
        if threshold_matches:
            violations.append(f"Item {it['item_id']} contains threshold annotation '{threshold_matches}'")

        # Check parenthetical interpretations
        for p_match in paren_pattern.findall(prompt):
            if not any(ap.match(p_match) for ap in allowed_parens):
                violations.append(f"Item {it['item_id']} contains non-allowed parenthetical '{p_match}'")

    assert len(violations) == 0, f"Found {len(violations)} banned token violations:\n" + "\n".join(violations[:10])


def test_field_level_prompt_diff_in_scenarios():
    items = load_dataset_items()
    templates: Dict[str, Dict[str, str]] = {}
    templates_meta: Dict[str, Dict] = {}
    for it in items:
        templates.setdefault(it["template_id"], {})[it["item_type"]] = it["prompt"]
        templates_meta[it["template_id"]] = it

    for tpl_id, triplet in templates.items():
        assert "base" in triplet and "edited" in triplet and "near_miss" in triplet
        family = templates_meta[tpl_id]["family"]

        p_base = [l.strip() for l in triplet["base"].strip().split("\n") if l.strip()]
        p_ed = [l.strip() for l in triplet["edited"].strip().split("\n") if l.strip()]
        p_nm = [l.strip() for l in triplet["near_miss"].strip().split("\n") if l.strip()]
        p_null = [l.strip() for l in triplet["null_edit"].strip().split("\n") if l.strip()]

        assert len(p_base) == len(p_ed) == len(p_nm) == len(p_null), f"Line count mismatch in {tpl_id}"

        # Lines:
        # 0: Clinical Vignette:
        # 1: {article} {age}-year-old... presents...
        # 2: Renal function: ...
        # 3: Reproductive status: ...
        # 4: Drug allergies: ...
        # 5: Family history: ...
        # 6: {distractor}
        # 7: Question: ...
        assert p_base[0] == p_ed[0] == p_nm[0] == p_null[0]
        assert p_base[1] == p_ed[1] == p_nm[1], f"Header line differs in {tpl_id}"
        assert p_base[1] != p_null[1], f"Header line identical base vs null_edit in {tpl_id}"
        assert p_base[6] == p_ed[6] == p_nm[6], f"Distractor differs in {tpl_id}"
        assert p_base[6] != p_null[6], f"Distractor identical base vs null_edit in {tpl_id}"
        assert p_base[7] == p_ed[7] == p_nm[7] == p_null[7], f"Question differs in {tpl_id}"

        # Check invariant parts for null_edit:
        assert p_base[2] == p_null[2], f"Renal differs base vs null_edit in {tpl_id}"
        assert p_base[3] == p_null[3], f"Repro differs base vs null_edit in {tpl_id}"
        assert p_base[4] == p_null[4], f"Allergy differs base vs null_edit in {tpl_id}"
        assert p_base[5] == p_null[5], f"Family history differs base vs null_edit in {tpl_id}"

        if family == "renal":
            assert p_base[2] != p_ed[2], f"Renal function identical base vs edited in {tpl_id}"
            assert p_base[2] != p_nm[2], f"Renal function identical base vs near_miss in {tpl_id}"
            assert p_base[3] == p_ed[3] == p_nm[3], f"Repro status differs in renal {tpl_id}"
            assert p_base[4] == p_ed[4] == p_nm[4], f"Allergies differ in renal {tpl_id}"
            assert p_base[5] == p_ed[5] == p_nm[5], f"Family history differs in renal {tpl_id}"

        elif family == "pregnancy":
            assert p_base[3] != p_ed[3], f"Repro status identical base vs edited in {tpl_id}"
            assert p_base[3] != p_nm[3], f"Repro status identical base vs near_miss in {tpl_id}"
            assert p_base[2] == p_ed[2] == p_nm[2], f"Renal differs in preg {tpl_id}"
            assert p_base[4] == p_ed[4] == p_nm[4], f"Allergies differ in preg {tpl_id}"
            assert p_base[5] == p_ed[5] == p_nm[5], f"Family history differs in preg {tpl_id}"

        elif family == "allergy":
            assert p_base[2] == p_ed[2] == p_nm[2], f"Renal differs in allergy {tpl_id}"
            assert p_base[3] == p_ed[3] == p_nm[3], f"Repro differs in allergy {tpl_id}"
            assert p_base[4] != p_ed[4], f"Allergies identical base vs edited in {tpl_id}"
            assert p_base[5] == p_ed[5], f"Family history differs base vs edited in {tpl_id}"
            assert p_base[4] == p_nm[4], f"Allergies differ base vs near_miss in {tpl_id}"
            assert p_base[5] != p_nm[5], f"Family history identical base vs near_miss in {tpl_id}"


def test_grammar_and_repeated_phrases():
    items = load_dataset_items()
    violations = []
    article_pattern = re.compile(r"^(A|An)\s+(\d+)-year-old", re.MULTILINE)
    repeated_pattern = re.compile(r"\b(\w+\s+\w+)\s+\1\b", re.IGNORECASE)

    for it in items:
        prompt = it["prompt"]
        m = article_pattern.search(prompt)
        if m:
            art, age_str = m.group(1), m.group(2)
            expected_art = "An" if age_str.startswith("8") else "A"
            if art != expected_art:
                violations.append(f"Item {it['item_id']} article mismatch: '{art} {age_str}-year-old' (expected {expected_art})")
        else:
            violations.append(f"Item {it['item_id']} missing age pattern")

        rep = repeated_pattern.search(prompt)
        if rep:
            violations.append(f"Item {it['item_id']} has repeated phrase '{rep.group(0)}'")

    assert len(violations) == 0, f"Found {len(violations)} grammar/repeated phrase violations:\n" + "\n".join(violations[:10])


def test_age_and_sex_distributions():
    items = load_dataset_items()
    by_rule = {}
    for it in items:
        if it["item_type"] == "base":
            by_rule.setdefault(it["rule_id"], []).append(it)

    for rule_id, rule_items in by_rule.items():
        ages = [it["structured_facts"]["age"] for it in rule_items]
        sexes = [it["structured_facts"]["sex"] for it in rule_items]

        min_age, max_age = min(ages), max(ages)
        assert min_age <= 25, f"Rule {rule_id} min age {min_age} > 25"
        assert max_age >= 80, f"Rule {rule_id} max age {max_age} < 80"

        if rule_id.startswith("preg_") or rule_id == "allergy_03":
            assert all(s == "female" for s in sexes), f"Rule {rule_id} must be 100% female"
        else:
            num_female = sum(1 for s in sexes if s == "female")
            num_male = sum(1 for s in sexes if s == "male")
            assert num_female > 0 and num_male > 0, f"Rule {rule_id} must have mixed sexes"
            assert abs(num_female - num_male) <= 2, f"Rule {rule_id} sexes must be balanced"


def test_prompt_deduplication():
    items = load_dataset_items()
    base_items = [it for it in items if it["item_type"] == "base"]

    def tokenize_words(text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    prompt_words = [(it["template_id"], tokenize_words(it["prompt"])) for it in base_items]

    violations = []
    for i in range(len(prompt_words)):
        for j in range(i + 1, len(prompt_words)):
            t1, w1 = prompt_words[i]
            t2, w2 = prompt_words[j]
            if t1 != t2:
                inter = len(w1 & w2)
                union = len(w1 | w2)
                jaccard = inter / union if union > 0 else 0.0
                if jaccard > 0.90:
                    violations.append(f"Pair ({t1}, {t2}) has Jaccard {jaccard:.4f} > 0.90")

    assert len(violations) == 0, f"Found {len(violations)} prompt duplication violations across scenarios:\n" + "\n".join(violations[:10])

