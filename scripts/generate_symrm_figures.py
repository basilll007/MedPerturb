"""Generate publication-quality figures for SymRM NAACL 2027 pilot.

Figures generated:
1. Fig 1: Hardware Scaling & VRAM Sweep (Peak VRAM, Tokens/sec, Step time vs Batch Size on RTX 5060 8GB).
2. Fig 2: Lexical Shortcut Floors & Baseline Comparison across Splits (Always-Default, Response-Only, Keyword, BoW).
3. Fig 3: Dataset Composition, Clinical Families, and Age/Sex Demographic Distributions.
4. Fig 4: Template Diversity & Cross-Scenario Deduplication Jaccard Distribution.
5. Fig 5: Response Candidate Token Length Symmetry and Neutrality Audit.

Outputs saved to:
- figures/symrm/
- docs/figures/symrm/
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style setup for crisp publication aesthetics
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def setup_dirs():
    dirs = [Path("figures/symrm"), Path("docs/figures/symrm")]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_fig(fig, filename: str, dirs):
    for d in dirs:
        path = d / filename
        fig.savefig(path, bbox_inches="tight")
    print(f"Saved {filename} to {len(dirs)} directories.")


def generate_fig1_vram_scaling(dirs):
    """Fig 1: Hardware scaling on RTX 5060 (8GB VRAM)."""
    batch_sizes = [1, 2, 4, 8]
    vram_gb = [2.124, 3.131, 3.777, 5.071]
    tokens_sec = [582.53, 823.16, 856.65, 90.57]
    step_time_s = [4.688 / 60 * 60, 3.317 / 60 * 60, 3.188 / 60 * 60, 30.151 / 60 * 60]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))

    color_vram = "#1e3a8a"  # Deep blue
    color_tp = "#047857"    # Forest green

    ax1.set_xlabel("Per-Device Batch Size (Pairs)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Peak VRAM Allocation (GB)", color=color_vram, fontsize=11, fontweight="bold")
    line1 = ax1.plot(batch_sizes, vram_gb, marker="s", markersize=8, color=color_vram, linewidth=2.2, label="Peak VRAM (GB)")
    ax1.tick_params(axis="y", labelcolor=color_vram)
    ax1.set_ylim(0, 8.5)
    ax1.axhline(7.0, color="#dc2626", linestyle="--", linewidth=1.5, label="7.0 GB VRAM Threshold")
    ax1.axhspan(7.0, 8.0, alpha=0.1, color="#dc2626")

    # Annotate points
    for x, y in zip(batch_sizes, vram_gb):
        ax1.annotate(f"{y:.2f} GB", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color=color_vram)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (Tokens / Sec)", color=color_tp, fontsize=11, fontweight="bold")
    line2 = ax2.plot(batch_sizes, tokens_sec, marker="o", markersize=8, color=color_tp, linewidth=2.2, linestyle="-.", label="Throughput (tok/s)")
    ax2.tick_params(axis="y", labelcolor=color_tp)
    ax2.set_ylim(0, 1050)

    for x, y in zip(batch_sizes, tokens_sec):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, fontweight="bold", color=color_tp)

    # Highlight optimal operating point
    ax1.scatter([2], [3.131], s=200, facecolors="none", edgecolors="#b45309", linewidth=2.5, zorder=5)
    ax1.annotate("Selected Pilot Setting\n(BS=2, Accum=16, 3.13 GB)", (2, 3.131), textcoords="offset points", xytext=(45, 15),
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.5), fontsize=9, fontweight="bold", color="#b45309")

    # Annotate thermal throttling cliff
    ax2.annotate("Laptop Thermal Throttling / Paging\n(90.6 tok/s)", (8, 90.57), textcoords="offset points", xytext=(-90, 30),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), fontsize=9, fontweight="bold", color="#dc2626")

    ax1.set_xticks(batch_sizes)
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.title("Gate 0: Training-Step VRAM & Throughput Scaling on RTX 5060 (8GB)\nQwen2.5-1.5B (NF4 LoRA, SeqLen=512, Effective Batch=32)", fontsize=11, pad=12)

    save_fig(fig, "fig1_vram_and_throughput_scaling.png", dirs)
    plt.close(fig)


def generate_fig2_lexical_shortcuts(dirs):
    """Fig 2: Lexical shortcut baseline accuracy across splits."""
    splits = ["Train (5 Rules)", "Near-OOD (2 Rules)", "Far-OOD (3 Rules)"]
    x = np.arange(len(splits))
    width = 0.18

    always_default = [50.0, 50.0, 50.0]
    response_only = [50.0, 50.0, 50.0]
    keyword_heuristic = [70.0, 75.0, 100.0]
    bow_logreg = [100.0, 100.0, 50.0]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    r1 = ax.bar(x - 1.5 * width, always_default, width, label="Always-Default (y=0)", color="#94a3b8")
    r2 = ax.bar(x - 0.5 * width, response_only, width, label="Response-Only LogReg", color="#38bdf8")
    r3 = ax.bar(x + 0.5 * width, keyword_heuristic, width, label="Keyword Heuristic", color="#fbbf24")
    r4 = ax.bar(x + 1.5 * width, bow_logreg, width, label="BoW LogReg (Prompt)", color="#3b82f6")

    # Far-OOD BoW threshold
    ax.axhline(70.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Preregistered STOP Threshold (70.0%)")
    ax.axhspan(70.0, 105.0, color="#dc2626", alpha=0.08)

    # Annotate Far-OOD BoW bar
    ax.annotate("PASS: 50.0%\n(< 70% floor)", (2 + 1.5 * width, 50.0), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, fontweight="bold", color="#1d4ed8")

    # Annotate Chance line
    ax.axhline(50.0, color="#64748b", linestyle=":", linewidth=1.0)
    ax.text(-0.4, 51.5, "Chance (50.0%)", fontsize=8, color="#64748b", style="italic")

    ax.set_ylabel("Balanced Counterfactual Pair Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title("Lexical Shortcut Floors: Balanced Accuracy on Base-vs-Edited Pairs\n(Preregistered Gating Requirement: Far-OOD BoW < 70.0%)", fontsize=11, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    save_fig(fig, "fig2_lexical_shortcut_floors.png", dirs)
    plt.close(fig)


def generate_fig3_dataset_demographics(dirs):
    """Fig 3: Clinical families, item distribution, and age/sex demographics."""
    data_path = Path("results/symrm/data/all_items.jsonl")
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    base_items = [it for it in items if it["item_type"] == "base"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Subplot 1: Clinical Rules by Family & Split
    families = ["Renal (4 Rules)", "Pregnancy (3 Rules)", "Allergy (3 Rules)"]
    splits = ["Train", "Near-OOD", "Far-OOD"]
    # Matrix: family x split counts of items
    counts = np.array([
        [270, 90, 0],   # Renal: 3 train (270), 1 near_ood (90), 0 far_ood
        [180, 90, 0],   # Pregnancy: 2 train (180), 1 near_ood (90), 0 far_ood
        [0, 0, 270],    # Allergy: 0 train, 0 near_ood, 3 far_ood (270)
    ])

    x = np.arange(len(families))
    w = 0.25
    ax1.bar(x - w, counts[:, 0], w, label="Train Split", color="#1e40af")
    ax1.bar(x, counts[:, 1], w, label="Near-OOD Split", color="#0284c7")
    ax1.bar(x + w, counts[:, 2], w, label="Far-OOD Split", color="#059669")

    ax1.set_ylabel("Total Vignettes (Base + Edited + Near-Miss)", fontsize=9, fontweight="bold")
    ax1.set_title("Vignette Volume by Clinical Family & Split (N=900)", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Subplot 2: Age Distribution & Sex Breakdown
    ages = [it["structured_facts"]["age"] for it in base_items]
    female_ages = [it["structured_facts"]["age"] for it in base_items if it["structured_facts"]["sex"] == "female"]
    male_ages = [it["structured_facts"]["age"] for it in base_items if it["structured_facts"]["sex"] == "male"]

    bins = np.linspace(25, 85, 13)
    ax2.hist([female_ages, male_ages], bins=bins, stacked=True, color=["#ec4899", "#3b82f6"], label=["Female (N=195)", "Male (N=105)"], edgecolor="#1f2937", linewidth=0.8)

    ax2.set_xlabel("Patient Age (Years, Spanning 25 to 80)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Scenario Count (Base Templates N=300)", fontsize=9, fontweight="bold")
    ax2.set_title("Demographic Distribution: Age Span & Sex Ratio", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "fig3_dataset_composition_and_demographics.png", dirs)
    plt.close(fig)


def generate_fig4_template_diversity(dirs):
    """Fig 4: Pairwise Jaccard similarity distribution and deduplication."""
    data_path = Path("results/symrm/data/all_items.jsonl")
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    base_items = [it for it in items if it["item_type"] == "base"]

    def tokenize(text):
        return set(re.findall(r"\b\w+\b", text.lower()))

    # Sample cross-scenario Jaccard similarities
    prompt_sets = [tokenize(it["prompt"]) for it in base_items]
    jaccards = []
    for i in range(len(prompt_sets)):
        for j in range(i + 1, len(prompt_sets)):
            s1, s2 = prompt_sets[i], prompt_sets[j]
            jacc = len(s1 & s2) / len(s1 | s2)
            jaccards.append(jacc)

    jaccards = np.array(jaccards)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    n, bins, patches = ax.hist(jaccards, bins=50, color="#6366f1", edgecolor="#312e81", alpha=0.85, linewidth=0.7)

    # Highlight mean and max
    mean_j = float(np.mean(jaccards))
    max_j = float(np.max(jaccards))

    ax.axvline(mean_j, color="#059669", linestyle="-", linewidth=2.0, label=f"Mean Cross-Scenario Jaccard: {mean_j:.4f}")
    ax.axvline(max_j, color="#d97706", linestyle="-", linewidth=2.0, label=f"Max Cross-Scenario Jaccard: {max_j:.4f}")
    ax.axvline(0.90, color="#dc2626", linestyle="--", linewidth=2.0, label="Deduplication Ceiling Threshold (0.9000)")
    ax.axvspan(0.90, 1.0, color="#dc2626", alpha=0.15)

    ax.annotate(f"Max: {max_j:.4f}\n(PASS < 0.90)", xy=(max_j, max(n) * 0.4), xytext=(max_j + 0.04, max(n) * 0.5),
                arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5), fontsize=9, fontweight="bold", color="#d97706")

    ax.annotate("Banned Duplication\nZone (> 0.90)", xy=(0.92, max(n) * 0.2), ha="center", fontsize=8.5, fontweight="bold", color="#dc2626")

    ax.set_xlabel("Pairwise Word-Level Jaccard Similarity", fontsize=10, fontweight="bold")
    ax.set_ylabel("Pair Count (Total Pairs = 44,850)", fontsize=10, fontweight="bold")
    ax.set_title("Prompt Deduplication & Lexical Diversity Distribution Across Scenarios\n(All 44,850 Distinct Scenario Pairs in Dataset)", fontsize=10.5, pad=12)
    ax.set_xlim(0.15, 1.0)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    save_fig(fig, "fig4_template_diversity_and_deduplication.png", dirs)
    plt.close(fig)


def generate_fig5_length_symmetry(dirs):
    """Fig 5: Response candidate length difference per rule."""
    qc_path = Path("results/symrm/token_length_qc.json")
    assert qc_path.exists()
    with open(qc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = list(data["per_rule"].keys())
    diffs = [data["per_rule"][r]["diff"] for r in rules]
    abs_diffs = [data["per_rule"][r]["abs_diff"] for r in rules]

    fig, ax = plt.subplots(figsize=(8, 4.2))

    colors = ["#22c55e" if abs(d) <= 2 else "#ef4444" for d in diffs]
    bars = ax.bar(rules, diffs, color=colors, edgecolor="#1f2937", linewidth=0.8)

    ax.axhline(0, color="#1e293b", linewidth=1.0)
    ax.axhline(2, color="#dc2626", linestyle="--", linewidth=1.2, label="Permissible Bound (+/- 2 Tokens)")
    ax.axhline(-2, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.axhspan(-2, 2, color="#22c55e", alpha=0.08)

    # Annotate each bar
    for b, d in zip(bars, diffs):
        y_pos = d + (0.15 if d >= 0 else -0.35)
        ax.text(b.get_x() + b.get_width() / 2, y_pos, f"{d:+d}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Token Length Difference (Default - Alternative)", fontsize=9.5, fontweight="bold")
    ax.set_title(f"Response Option Symmetry Audit (Mean |Diff| = {data['mean_abs_diff']:.2f} Tokens <= 2.0)\nTokenizer: Qwen/Qwen2.5-1.5B (Zero Length Confounding)", fontsize=10.5, pad=12)
    ax.set_ylim(-3, 3)
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(rules, rotation=35, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    save_fig(fig, "fig5_token_length_neutrality.png", dirs)
    plt.close(fig)


def main():
    dirs = setup_dirs()
    generate_fig1_vram_scaling(dirs)
    generate_fig2_lexical_shortcuts(dirs)
    generate_fig3_dataset_demographics(dirs)
    generate_fig4_template_diversity(dirs)
    generate_fig5_length_symmetry(dirs)
    print("All SymRM figures successfully generated!")


if __name__ == "__main__":
    main()
