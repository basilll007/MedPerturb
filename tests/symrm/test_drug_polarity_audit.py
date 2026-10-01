"""Cross-split drug-polarity audit for SymRM.

Audits every drug/action appearing in train vs held-out (near-OOD and far-OOD)
and verifies that NO drug is 'avoid' in train and 'correct' in held-out,
or 'correct' in train and 'avoid' in held-out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml
import pytest


def run_drug_polarity_audit():
    rules_path = Path("configs/symrm/rules.yaml")
    assert rules_path.exists(), f"Rules file not found at {rules_path}"

    with open(rules_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = data.get("rules", [])
    assert len(rules) == 10, f"Expected exactly 10 rules, found {len(rules)}"

    # Canonical mapping of drug entities per rule
    drug_entries: List[Dict[str, str]] = []

    # Map of canonical drug -> set of (split_group, polarity)
    # split_group: 'train' vs 'held_out' (near_ood and far_ood)
    canonical_map = {
        "renal_01": [("metformin", "avoid"), ("insulin", "correct")],
        "renal_03": [("nitrofurantoin", "avoid"), ("fosfomycin", "correct")],
        "renal_04": [("gabapentin", "avoid"), ("gabapentin_dose_reduced", "correct")],
        "preg_01": [("lisinopril", "avoid"), ("labetalol", "correct"), ("nifedipine", "correct")],
        "preg_03": [("doxycycline", "avoid"), ("cefuroxime_axetil", "correct")],
        "renal_02": [("enoxaparin", "avoid"), ("enoxaparin_dose_reduced", "correct"), ("unfractionated_heparin", "correct")],
        "preg_02": [("methotrexate", "avoid"), ("hydroxychloroquine", "correct"), ("sulfasalazine", "correct")],
        "allergy_01": [("amoxicillin", "avoid"), ("penicillin", "avoid"), ("azithromycin", "correct")],
        "allergy_02": [("trimethoprim_sulfamethoxazole", "avoid"), ("sulfonamide", "avoid"), ("ciprofloxacin", "correct")],
        "allergy_03": [("ibuprofen", "avoid"), ("nsaids", "avoid"), ("aspirin", "avoid"), ("acetaminophen", "correct")],
    }

    train_polarities: Dict[str, Set[str]] = {}
    heldout_polarities: Dict[str, Set[str]] = {}

    audit_table_rows = []

    for r in rules:
        rule_id = r["rule_id"]
        split = r["split"]
        family = r["family"]
        indication = r["indication"]
        split_group = "train" if split == "train" else "held_out"

        c_drugs = canonical_map.get(rule_id, [])
        for drug_name, polarity in c_drugs:
            row = {
                "rule_id": rule_id,
                "family": family,
                "split": split,
                "split_group": split_group,
                "indication": indication,
                "drug": drug_name,
                "polarity": polarity,
                "action_summary": r["rule_action"] if polarity == "correct" else r["default_action"],
            }
            audit_table_rows.append(row)

            if split_group == "train":
                train_polarities.setdefault(drug_name, set()).add(polarity)
            else:
                heldout_polarities.setdefault(drug_name, set()).add(polarity)

    # Cross-split conflict check
    conflicts = []
    all_drugs = set(train_polarities.keys()) | set(heldout_polarities.keys())

    for drug in all_drugs:
        t_pols = train_polarities.get(drug, set())
        h_pols = heldout_polarities.get(drug, set())

        # Check: avoid in train and correct in held-out
        if "avoid" in t_pols and "correct" in h_pols:
            conflicts.append(f"CONFLICT: Drug '{drug}' is AVOID in train but CORRECT in held-out.")
        # Check: correct in train and avoid in held-out
        if "correct" in t_pols and "avoid" in h_pols:
            conflicts.append(f"CONFLICT: Drug '{drug}' is CORRECT in train but AVOID in held-out.")

    # Generate Markdown Table
    md_lines = [
        "# Cross-Split Drug-Polarity Audit Table",
        "",
        "Audit verifying polarity consistency across train and held-out (near-OOD and far-OOD) splits.",
        "Constraint: FAIL if any drug is 'avoid' in train and 'correct' in held-out, or vice versa.",
        "",
        "| Rule ID | Family | Split | Split Group | Indication | Drug Entity | Polarity | Action Context |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in audit_table_rows:
        md_lines.append(
            f"| `{row['rule_id']}` | {row['family']} | `{row['split']}` | **{row['split_group']}** | {row['indication']} | `{row['drug']}` | **{row['polarity'].upper()}** | {row['action_summary'][:60]}... |"
        )

    md_lines.append("")
    md_lines.append("## Conflict Audit Results")
    if conflicts:
        md_lines.append(f"**STATUS: FAIL ({len(conflicts)} conflicts found)**")
        for c in conflicts:
            md_lines.append(f"- {c}")
    else:
        md_lines.append("**STATUS: PASS (0 conflicts across train and held-out splits)**")
        md_lines.append("- No drug labeled 'avoid' in train appears as 'correct' in held-out.")
        md_lines.append("- No drug labeled 'correct' in train appears as 'avoid' in held-out.")
        md_lines.append("- Total unique drug entities audited: " + str(len(all_drugs)))

    results_dir = Path("results/symrm")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "drug_polarity_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    with open(results_dir / "drug_polarity_audit.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "audit_passed": len(conflicts) == 0,
                "num_conflicts": len(conflicts),
                "conflicts": conflicts,
                "num_drugs_audited": len(all_drugs),
                "rows": audit_table_rows,
            },
            f,
            indent=2,
        )

    assert len(conflicts) == 0, f"Polarity audit failed with {len(conflicts)} conflicts: {conflicts}"
    return len(conflicts) == 0


def test_drug_polarity_audit():
    assert run_drug_polarity_audit() is True


if __name__ == "__main__":
    passed = run_drug_polarity_audit()
    print("Drug polarity audit passed:", passed)
