"""Phase 3 — Manuscript Figures and Tables Generator.

Reads saved evaluation artifacts (responses.csv, paired_results.csv, metrics_by_perturbation.csv)
and generates publication-ready figures (300 DPI PNG + PDF) and LaTeX/CSV tables.

Never hard-codes any experimental results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medperturb.evaluation.paired_stats import compute_paired_comparison_report

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures" / "phase3"
TABLES_DIR = ROOT / "tables" / "phase3"

# Styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "Base": "#6c757d",           # Neutral Slate Gray
    "Correctness": "#0d6efd",    # Primary Blue
    "Neuro-Symbolic": "#198754", # Emerald Green
}


def generate_pipeline_figure(out_path: Path) -> None:
    """Generates Figure 1: Pipeline architecture diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")

    boxes = [
        ("ReMedQA\nBenchmark\n(MedQA, MedMCQA,\nMMLU-Clinical)", (0.10, 0.5), "#f8f9fa", "#212529"),
        ("Symbolic\nTransformations\n(Invariance &\nAdaptation)", (0.32, 0.5), "#e7f1ff", "#0d6efd"),
        ("Verifiable\nReward Engine\n$R_{\\text{NS}} = R_{\\text{corr}} + \\lambda R_{\\text{symb}}$", (0.55, 0.5), "#e8f5e9", "#198754"),
        ("Post-Training\nGRPO + LoRA\n(Unsloth 4-bit\nRTX 5060 8GB)", (0.76, 0.5), "#fff3cd", "#ffc107"),
        ("Paired Audit &\nFinal Evaluation\n(Zero Leakage\nUntouched Test)", (0.94, 0.5), "#f8d7da", "#dc3545"),
    ]

    for title, (x, y), face_col, edge_col in boxes:
        bbox_props = dict(boxstyle="round,pad=0.6,rounding_size=0.2", facecolor=face_col, edgecolor=edge_col, linewidth=1.5)
        ax.text(x, y, title, ha="center", va="center", bbox=bbox_props, fontsize=9.5, fontweight="bold", wrap=True)

    # Draw arrows between boxes
    arrow_props = dict(facecolor="#495057", edgecolor="#495057", width=1.5, headwidth=7, shrink=0.1)
    ax.annotate("", xy=(0.22, 0.5), xytext=(0.18, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.43, 0.5), xytext=(0.40, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.67, 0.5), xytext=(0.63, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.84, 0.5), xytext=(0.82, 0.5), arrowprops=arrow_props)

    ax.set_title("MedPerturb Neuro-Symbolic Post-Training Pipeline", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[+] Saved pipeline figure to {out_path}")


def generate_adaptation_failure_figure(
    models_data: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Generates Figure 2: Primary Outcome — Required Adaptation Failure Rate Comparison."""
    adapt_conds = ["none_of_the_provided", "incorrect"]
    rates, errors, labels, colors = [], [], [], []

    for name, paired_df in models_data.items():
        sub = paired_df[paired_df["condition"].isin(adapt_conds)]
        fails = sub["adaptation"].fillna("").str.startswith("failure").to_numpy(dtype=float)
        mean_rate = np.mean(fails) * 100.0
        # Standard error of the mean
        sem = (np.std(fails) / np.sqrt(len(fails))) * 100.0
        rates.append(mean_rate)
        errors.append(sem)
        labels.append(name)
        colors.append(COLORS.get(name, "#495057"))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, yerr=errors, capsize=6, color=colors, alpha=0.9, edgecolor="black", linewidth=1.2, width=0.55)

    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.8, f"{r:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10.5)

    ax.set_ylabel("Required Adaptation Failure Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Primary Outcome: Required Adaptation Failure Rate\n(Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[+] Saved adaptation failure figure to {out_path}")


def generate_per_perturbation_figure(
    models_metrics: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Generates Figure 3: Per-Perturbation Performance across models."""
    display_conds = ["mcq", "roman_numeral", "fixed_pos", "no_symbols", "none_of_the_provided", "incorrect"]
    cond_labels = ["MCQ\n(Baseline)", "Roman\n(Invariance)", "Fixed Pos\n(Invariance)", "No Symbols\n(Invariance)", "None of Provided\n(Adaptation)", "Incorrect Set\n(Inversion)"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    n_models = len(models_metrics)
    width = 0.8 / n_models
    x = np.arange(len(display_conds))

    for idx, (model_name, metrics_df) in enumerate(models_metrics.items()):
        sub = metrics_df.set_index("condition")
        accs = [sub.loc[c, "accuracy"] * 100.0 if c in sub.index and not pd.isna(sub.loc[c, "accuracy"]) else 0.0 for c in display_conds]
        offset = (idx - (n_models - 1) / 2.0) * width
        rects = ax.bar(x + offset, accs, width, label=model_name, color=COLORS.get(model_name, "#333"), alpha=0.9, edgecolor="black", linewidth=1.0)

    ax.set_ylabel("Task Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Across Perturbation Conditions", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[+] Saved per-perturbation figure to {out_path}")


def generate_invariance_adaptation_tradeoff_figure(
    models_paired: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Generates Figure 4: Invariance Accuracy vs Adaptation Success Tradeoff."""
    invar_conds = ["roman_numeral", "fixed_pos", "no_symbols"]
    adapt_conds = ["none_of_the_provided", "incorrect"]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for name, paired_df in models_paired.items():
        invar_acc = paired_df[paired_df["condition"].isin(invar_conds)]["perturbed_correct"].mean() * 100.0
        adapt_sub = paired_df[paired_df["condition"].isin(adapt_conds)]
        adapt_succ = (adapt_sub["adaptation"] == "successful_adaptation").mean() * 100.0

        ax.scatter(invar_acc, adapt_succ, s=180, label=name, color=COLORS.get(name, "#333"), edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(name, (invar_acc, adapt_succ), textcoords="offset points", xytext=(8, 5), fontweight="bold", fontsize=10)

    ax.set_xlabel("Invariance Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Adaptation Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Invariance vs Adaptation Tradeoff Space\n(Top-Right is Optimal)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[+] Saved tradeoff figure to {out_path}")


def generate_latex_and_csv_tables(
    models_resp: dict[str, pd.DataFrame],
    models_paired: dict[str, pd.DataFrame],
    models_metrics: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Generates Table 1 (Main comparison) and Table 2 (Per-perturbation) in LaTeX and CSV."""
    invar_conds = ["roman_numeral", "fixed_pos", "no_symbols"]
    adapt_conds = ["none_of_the_provided", "incorrect"]

    rows = []
    for name in models_resp.keys():
        resp_df = models_resp[name]
        paired_df = models_paired[name]

        mcq_acc = resp_df[resp_df["condition"] == "mcq"]["correct"].mean() * 100.0
        invar_acc = paired_df[paired_df["condition"].isin(invar_conds)]["perturbed_correct"].mean() * 100.0
        adapt_acc = paired_df[paired_df["condition"].isin(adapt_conds)]["perturbed_correct"].mean() * 100.0
        adapt_fail = (paired_df[paired_df["condition"].isin(adapt_conds)]["adaptation"].fillna("").str.startswith("failure")).mean() * 100.0

        # Joint consistency on invariance
        by_q = resp_df[resp_df["condition"].isin(["mcq"] + invar_conds)].groupby("question_id")
        reacc = by_q["correct"].apply(lambda s: s.all()).mean() * 100.0

        rows.append({
            "Model Policy": name,
            "MCQ Acc (%)": round(mcq_acc, 2),
            "Invariance Acc (%)": round(invar_acc, 2),
            "ReAcc (%)": round(reacc, 2),
            "Adaptation Acc (%)": round(adapt_acc, 2),
            "Adaptation Failure Rate (%) [Primary]": round(adapt_fail, 2),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "table1_main_results.csv", index=False)

    latex_str = summary_df.to_latex(index=False, caption="MedPerturb Experimental Evaluation: Primary and Secondary Outcomes across Model Policies.", label="tab:medperturb_main", float_format="%.2f")
    with open(out_dir / "table1_main_results.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"[+] Saved Table 1 to {out_dir}")
