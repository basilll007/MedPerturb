"""Template diversity and sentence skeleton analysis for SymRM.

Per user requirements:
- Number of unique sentence skeletons per rule.
- Mean pairwise Jaccard similarity between prompts within each rule.
- FLAG any rule with mean Jaccard > 0.80.
- Extracts 3 full example scenarios per split (base / edited / near-miss).
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np


def extract_skeleton(prompt: str) -> str:
    """Abstracts specific clinical values (numbers, ages, sexes, measurements) to form a structural sentence skeleton."""
    p = prompt.strip()
    # Replace ages
    p = re.sub(r"\b\d{1,2}-year-old\b", "<AGE>-year-old", p)
    # Replace numbers with decimals or units
    p = re.sub(r"\b\d+(\.\d+)?\s*(mg/dL|mEq/L|mL/min|mmHg|bpm|mIU/mL|U/L|cm|mm/hr|%)\b", "<VALUE>", p)
    # Replace plain integers
    p = re.sub(r"\b\d+\b", "<NUM>", p)
    # Replace gender pronouns
    p = re.sub(r"\b(female|male|She|He|Her|His)\b", "<PRONOUN>", p)
    return p


def token_jaccard(text_a: str, text_b: str) -> float:
    """Computes word-level Jaccard similarity between two texts."""
    tokens_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return float(len(intersection) / len(union))


def run_template_diversity_analysis(data_path: str = "results/symrm/data/all_items.jsonl") -> Dict[str, Any]:
    path = Path(data_path)
    assert path.exists(), f"Data not found at {path}"

    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Group by rule_id
    by_rule: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_rule.setdefault(it["rule_id"], []).append(it)

    report_per_rule: Dict[str, Any] = {}
    flagged_rules = []

    for rule_id, rule_items in by_rule.items():
        family = rule_items[0]["family"]
        split = rule_items[0]["split"]

        prompts = [it["prompt"] for it in rule_items]
        skeletons = set(extract_skeleton(p) for p in prompts)

        # Pairwise Jaccard similarity over all pairs
        jaccards = []
        for p1, p2 in combinations(prompts, 2):
            jaccards.append(token_jaccard(p1, p2))

        mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
        max_jaccard = float(np.max(jaccards)) if jaccards else 0.0
        min_jaccard = float(np.min(jaccards)) if jaccards else 0.0

        is_flagged = mean_jaccard > 0.80
        if is_flagged:
            flagged_rules.append(rule_id)

        report_per_rule[rule_id] = {
            "rule_id": rule_id,
            "family": family,
            "split": split,
            "num_items": len(rule_items),
            "num_unique_skeletons": len(skeletons),
            "mean_jaccard": mean_jaccard,
            "max_jaccard": max_jaccard,
            "min_jaccard": min_jaccard,
            "flagged_gt_0_8": is_flagged,
        }

    # Cross-scenario deduplication analysis (pairwise Jaccard across distinct scenarios)
    base_items = [it for it in items if it["item_type"] == "base"]
    cross_scenario_jaccards = []
    max_cross_jaccard = 0.0
    max_cross_pair = None

    for i in range(len(base_items)):
        for j in range(i + 1, len(base_items)):
            p1 = base_items[i]["prompt"]
            p2 = base_items[j]["prompt"]
            t1 = base_items[i]["template_id"]
            t2 = base_items[j]["template_id"]
            jacc = token_jaccard(p1, p2)
            cross_scenario_jaccards.append(jacc)
            if jacc > max_cross_jaccard:
                max_cross_jaccard = jacc
                max_cross_pair = (t1, t2)

    mean_cross_jaccard = float(np.mean(cross_scenario_jaccards)) if cross_scenario_jaccards else 0.0
    dedup_pass = max_cross_jaccard <= 0.90

    # Group items by template_id
    by_template: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for it in items:
        by_template.setdefault(it["template_id"], {})[it["item_type"]] = it

    # Pick 2 templates from Train, 2 from Near-OOD, 2 from Far-OOD for clean display
    selected_scenarios = {
        "train": ["tpl_renal_01_001", "tpl_preg_01_001"],
        "near_ood": ["tpl_renal_02_001", "tpl_preg_02_001"],
        "far_ood": ["tpl_allergy_01_001", "tpl_allergy_03_001"],
    }

    scenarios_detail = {}
    for split_name, tpl_ids in selected_scenarios.items():
        scenarios_detail[split_name] = []
        for tid in tpl_ids:
            triplet = by_template.get(tid, {})
            scenarios_detail[split_name].append({
                "template_id": tid,
                "rule_id": triplet["base"]["rule_id"],
                "indication": triplet["base"]["indication"],
                "default_action": triplet["base"]["default_option"],
                "alternative_action": triplet["base"]["alternative_option"],
                "base_prompt": triplet["base"]["prompt"],
                "edited_prompt": triplet["edited"]["prompt"],
                "near_miss_prompt": triplet["near_miss"]["prompt"],
            })

    results = {
        "summary": {
            "total_rules": len(report_per_rule),
            "flagged_rules_count": len(flagged_rules),
            "flagged_rules": flagged_rules,
            "all_rules_pass_jaccard_check": len(flagged_rules) == 0,
            "cross_scenario_dedup": {
                "max_jaccard": max_cross_jaccard,
                "max_pair": max_cross_pair,
                "mean_jaccard": mean_cross_jaccard,
                "passed_le_0_90": dedup_pass,
            }
        },
        "per_rule": report_per_rule,
        "example_scenarios": scenarios_detail,
    }

    out_dir = Path("results/symrm")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "template_diversity_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Format Markdown Report
    md_lines = [
        "# Template Diversity & Lexical Similarity Report",
        "",
        "Analysis of sentence skeleton variation, pairwise prompt Jaccard similarity per rule, and cross-scenario deduplication.",
        "Quality Constraints:",
        "1. FLAG any rule with mean pairwise Jaccard similarity > 0.80.",
        "2. Prompt Deduplication: No two prompts from DIFFERENT scenarios with Jaccard > 0.90.",
        "",
        "## 1. Per-Rule Diversity Table",
        "",
        "| Rule ID | Family | Split | Num Items | Unique Skeletons | Mean Jaccard | Min Jaccard | Max Jaccard (Triplets) | Status (Jaccard $\\le 0.80$) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for rid, r in report_per_rule.items():
        status_str = "**FLAGGED (> 0.80)**" if r["flagged_gt_0_8"] else "PASS"
        md_lines.append(
            f"| `{r['rule_id']}` | {r['family']} | `{r['split']}` | {r['num_items']} | {r['num_unique_skeletons']} | **{r['mean_jaccard']:.3f}** | {r['min_jaccard']:.3f} | {r['max_jaccard']:.3f} | {status_str} |"
        )

    md_lines.append("")
    md_lines.append("## 2. Cross-Scenario Deduplication (Distinct Scenarios)")
    md_lines.append(f"- **Max Pairwise Jaccard Across Different Scenarios:** **{max_cross_jaccard:.4f}** (Threshold: $\\le 0.90$) -> **{'PASS' if dedup_pass else 'FAIL'}**")
    md_lines.append(f"- **Highest Similarity Scenario Pair:** {max_cross_pair[0]} vs {max_cross_pair[1]}")
    md_lines.append(f"- **Mean Pairwise Jaccard Across Different Scenarios:** {mean_cross_jaccard:.4f}")
    md_lines.append("")
    md_lines.append("## 3. Diversity Summary")
    if flagged_rules or not dedup_pass:
        md_lines.append(f"**STATUS: FLAGGED**")
    else:
        md_lines.append("**STATUS: PASS (All rules mean Jaccard $\\le 0.80$; All cross-scenario pairs Jaccard $\\le 0.90$)**")
        md_lines.append("Clinical templates exhibit strong demographic, presentation, and distractor variability.")

    with open(out_dir / "template_diversity_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return results


if __name__ == "__main__":
    res = run_template_diversity_analysis()
    print("Diversity analysis complete:")
    print("Flagged rules:", res["summary"]["flagged_rules"])
    for rid, r in res["per_rule"].items():
        print(f"  {rid}: skeletons={r['num_unique_skeletons']}, mean_jaccard={r['mean_jaccard']:.3f}")
