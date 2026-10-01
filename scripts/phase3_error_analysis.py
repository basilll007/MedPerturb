"""Phase 3 — Structured Failure and Error Analysis.

Analyzes failure dynamics across:
- Base Qwen
- Correctness-trained Qwen
- Neuro-Symbolic Qwen

Extracts concrete qualitative case studies:
1. Base fails -> Correctness fails -> Neuro-Symbolic succeeds
2. Cases where Neuro-Symbolic training hurts performance
3. Taxonomy decomposition:
   - Knowledge errors
   - Representation instability
   - Adaptation failures
   - Instruction-following failures
   - Parse failures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def perform_error_analysis(
    base_dir: Path,
    correctness_dir: Path,
    neurosymbolic_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    base_resp = pd.read_csv(base_dir / "responses.csv")
    corr_resp = pd.read_csv(correctness_dir / "responses.csv")
    ns_resp = pd.read_csv(neurosymbolic_dir / "responses.csv")

    base_fail = pd.read_csv(base_dir / "failures.csv") if (base_dir / "failures.csv").exists() else pd.DataFrame()
    corr_fail = pd.read_csv(correctness_dir / "failures.csv") if (correctness_dir / "failures.csv").exists() else pd.DataFrame()
    ns_fail = pd.read_csv(neurosymbolic_dir / "failures.csv") if (neurosymbolic_dir / "failures.csv").exists() else pd.DataFrame()

    # Align by (question_id, condition)
    merged = pd.merge(
        base_resp[["question_id", "condition", "correct", "raw_response", "normalized_answer", "gold_option_text"]],
        corr_resp[["question_id", "condition", "correct", "raw_response", "normalized_answer"]],
        on=["question_id", "condition"],
        suffixes=("_base", "_corr"),
    )
    merged = pd.merge(
        merged,
        ns_resp[["question_id", "condition", "correct", "raw_response", "normalized_answer"]],
        on=["question_id", "condition"],
    )
    merged.rename(
        columns={
            "correct": "correct_ns",
            "raw_response": "raw_response_ns",
            "normalized_answer": "normalized_answer_ns",
        },
        inplace=True,
    )

    # 1. Triplet Category 1: Base fails -> Correctness fails -> NS succeeds
    ns_wins = merged[(merged["correct_base"] == False) & (merged["correct_corr"] == False) & (merged["correct_ns"] == True)]

    # 2. Triplet Category 2: NS hurts (Base OR Correctness succeeds, but NS fails)
    ns_hurts = merged[((merged["correct_base"] == True) | (merged["correct_corr"] == True)) & (merged["correct_ns"] == False)]

    # Failure taxonomy breakdown
    taxonomy = {
        "Base": base_fail["failure_type"].value_counts().to_dict() if not base_fail.empty else {},
        "Correctness": corr_fail["failure_type"].value_counts().to_dict() if not corr_fail.empty else {},
        "Neuro-Symbolic": ns_fail["failure_type"].value_counts().to_dict() if not ns_fail.empty else {},
    }

    # Save detailed case studies
    ns_wins_sample = ns_wins.head(20).to_dict(orient="records")
    ns_hurts_sample = ns_hurts.head(20).to_dict(orient="records")

    report = {
        "total_evaluated_pairs": len(merged),
        "ns_rescues_count": len(ns_wins),
        "ns_hurts_count": len(ns_hurts),
        "net_ns_benefit": len(ns_wins) - len(ns_hurts),
        "taxonomy": taxonomy,
        "ns_rescues_examples": ns_wins_sample,
        "ns_hurts_examples": ns_hurts_sample,
    }

    with open(out_dir / "error_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Generate Markdown summary
    md_lines = [
        "# MedPerturb Phase 3 — Qualitative & Quantitative Error Analysis\n",
        f"- **Total Evaluated Pairs**: {len(merged)}",
        f"- **Cases where NS Rescues (Base ✗ → Corr ✗ → NS ✓)**: {len(ns_wins)}",
        f"- **Cases where NS Hurts (Base/Corr ✓ → NS ✗)**: {len(ns_hurts)}",
        f"- **Net Semantic Benefit**: {len(ns_wins) - len(ns_hurts)} pairs\n",
        "## Failure Taxonomy Across Policies\n",
        pd.DataFrame(taxonomy).fillna(0).astype(int).to_markdown(),
        "\n## Exemplar Cases: Neuro-Symbolic Rescues\n",
    ]

    for ex in ns_wins_sample[:5]:
        md_lines.append(f"### Question `{ex['question_id']}` ({ex['condition']})")
        md_lines.append(f"- **Gold Target**: {ex.get('gold_option_text')}")
        md_lines.append(f"- **Base Prediction**: `{ex.get('normalized_answer_base')}` (Incorrect)")
        md_lines.append(f"- **Correctness Prediction**: `{ex.get('normalized_answer_corr')}` (Incorrect)")
        md_lines.append(f"- **Neuro-Symbolic Prediction**: `{ex.get('normalized_answer_ns')}` (CORRECT)")
        md_lines.append("")

    md_lines.append("## Exemplar Cases: Neuro-Symbolic Regressions (Honest Negative Cases)\n")
    for ex in ns_hurts_sample[:5]:
        md_lines.append(f"### Question `{ex['question_id']}` ({ex['condition']})")
        md_lines.append(f"- **Gold Target**: {ex.get('gold_option_text')}")
        md_lines.append(f"- **Base Prediction**: `{ex.get('normalized_answer_base')}`")
        md_lines.append(f"- **Correctness Prediction**: `{ex.get('normalized_answer_corr')}`")
        md_lines.append(f"- **Neuro-Symbolic Prediction**: `{ex.get('normalized_answer_ns')}` (INCORRECT)")
        md_lines.append("")

    with open(out_dir / "error_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Error analysis report generated at {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--correctness_dir", type=str, required=True)
    parser.add_argument("--neurosymbolic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    perform_error_analysis(
        Path(args.base_dir),
        Path(args.correctness_dir),
        Path(args.neurosymbolic_dir),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
