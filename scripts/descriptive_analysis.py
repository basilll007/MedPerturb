"""
Descriptive analysis of ReMedQA perturbations (Phase 0, Task 9).

Purely DESCRIPTIVE statistics over the pairing table built by
scripts/build_pairs.py. IMPORTANT: lexical similarity between original and
perturbed question text is a surface-level string-similarity signal only.
It is NOT interpreted anywhere in this script (or its outputs) as proof or
evidence of semantic equivalence between the original and perturbed
question. Two questions can be lexically dissimilar but semantically
equivalent (e.g. `open`'s reworded questions) or lexically identical but
task-semantically different (e.g. `incorrect`, which reuses the exact
question text but inverts what is being asked).

Lexical similarity metric used: Python stdlib `difflib.SequenceMatcher`
ratio (0..1) on lowercased, punctuation-stripped, whitespace-collapsed text
(same normalization as build_pairs.py's `question_text_changed` flag).
Chosen over token-set Jaccard because it is sensitive to word order, which
matters for detecting question rewording (Jaccard would score a shuffled
question as identical). Applied consistently to every perturbation_type.

Run standalone: uv run python scripts\\descriptive_analysis.py
Reads:  data/processed/remedqa_pairs.parquet
Writes: results/audit/perturbation_descriptives.csv
        figures/audit/perturbation_counts.png
        figures/audit/length_change_by_perturbation.png
        figures/audit/lexical_similarity_by_perturbation.png
        figures/audit/answer_position_distribution.png
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAIRS_PARQUET = PROJECT_ROOT / "data" / "processed" / "remedqa_pairs.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "audit"
FIGURES_DIR = PROJECT_ROOT / "figures" / "audit"
DESCRIPTIVES_CSV = RESULTS_DIR / "perturbation_descriptives.csv"

SOURCES = ["medqa", "medmcqa", "mmlu"]
PERTURBATIONS = ["open", "incorrect", "roman_numeral", "none_of_the_provided", "fixed_pos", "no_symbols"]
# Perturbations whose answer is a single key (letter or roman numeral)
# resolvable to a position within that row's own options dict — used for the
# answer-position / letter change analysis.
POSITION_COMPARABLE_PERTURBATIONS = ["roman_numeral", "none_of_the_provided", "fixed_pos"]
LETTER_LABELS = ["A", "B", "C", "D", "E", "F"]  # generous upper bound

# dataviz skill categorical palette (fixed order, validated colorblind-safe
# adjacent-pair separation) — reused as-is for static matplotlib figures.
CAT_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
    }
)


def normalize_text(s) -> str:
    if not isinstance(s, str):
        return ""
    out = s.lower().strip()
    out = "".join(ch for ch in out if ch.isalnum() or ch.isspace())
    return " ".join(out.split())


def word_count(s) -> int:
    if not isinstance(s, str):
        return 0
    return len(s.split())


def lexical_similarity(a: str, b: str) -> float:
    """Descriptive lexical similarity only (difflib SequenceMatcher ratio on
    normalized text). NOT a semantic-equivalence measure — see module
    docstring."""
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def options_avg_wordcount(options_json: str | None) -> float | None:
    if not options_json:
        return None
    try:
        opts = json.loads(options_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(opts, dict) or not opts:
        return None
    return float(np.mean([word_count(v) for v in opts.values()]))


def option_position(options_json: str | None, key) -> int | None:
    if not options_json or key is None:
        return None
    try:
        opts = json.loads(options_json)
    except (json.JSONDecodeError, TypeError):
        return None
    keys = list(opts.keys())
    if key not in keys:
        return None
    return keys.index(key)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["orig_word_count"] = df["original_question"].apply(word_count)
    df["pert_word_count"] = df["perturbed_question"].apply(word_count)
    df["question_length_diff"] = df["pert_word_count"] - df["orig_word_count"]

    df["orig_options_avg_wc"] = df["original_options"].apply(options_avg_wordcount)
    df["pert_options_avg_wc"] = df["perturbed_options"].apply(options_avg_wordcount)
    df["options_length_diff"] = df["pert_options_avg_wc"] - df["orig_options_avg_wc"]

    df["lexical_similarity"] = df.apply(
        lambda r: lexical_similarity(r["original_question"], r["perturbed_question"]), axis=1
    )

    # Answer-position analysis, only meaningful for perturbations whose
    # answer is a key into that row's own options dict.
    def orig_pos(row):
        if row["perturbation_type"] not in POSITION_COMPARABLE_PERTURBATIONS:
            return None
        return option_position(row["original_options"], row["original_answer_letter"])

    def pert_pos(row):
        if row["perturbation_type"] not in POSITION_COMPARABLE_PERTURBATIONS:
            return None
        return option_position(row["perturbed_options"], row["perturbed_answer_raw"])

    df["orig_answer_position"] = df.apply(orig_pos, axis=1)
    df["pert_answer_position"] = df.apply(pert_pos, axis=1)
    df["answer_position_changed"] = np.where(
        df["perturbation_type"].isin(POSITION_COMPARABLE_PERTURBATIONS),
        df["orig_answer_position"] != df["pert_answer_position"],
        np.nan,
    )
    # Literal raw-token comparison (letter string vs letter string /
    # roman-numeral string) — trivially near-100% "changed" for
    # roman_numeral since the token alphabet itself changes (A->I etc.);
    # reported for transparency/completeness alongside the more meaningful
    # position-index comparison above.
    df["raw_answer_token_changed"] = np.where(
        df["perturbation_type"].isin(POSITION_COMPARABLE_PERTURBATIONS),
        df["original_answer_letter"] != df["perturbed_answer_raw"],
        np.nan,
    )

    def answer_format(raw: str) -> str:
        if not isinstance(raw, str):
            return "unknown"
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            return "list_of_letters"
        if s in {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}:
            return "roman_numeral"
        if len(s) == 1 and s.isalpha() and s.isupper():
            return "letter"
        return "free_text"

    df["answer_format"] = df["perturbed_answer_raw"].apply(answer_format)

    return df


def build_descriptives_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pert in PERTURBATIONS:
        sub = df[df["perturbation_type"] == pert]
        n = len(sub)
        qlen_diff = sub["question_length_diff"]
        opt_diff = sub["options_length_diff"].dropna()
        lex_sim = sub["lexical_similarity"]
        pos_applicable = pert in POSITION_COMPARABLE_PERTURBATIONS
        pos_change_rate = (
            sub["answer_position_changed"].astype(float).mean() if pos_applicable else np.nan
        )
        raw_token_change_rate = (
            sub["raw_answer_token_changed"].astype(float).mean() if pos_applicable else np.nan
        )
        fmt_dist = sub["answer_format"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "perturbation_type": pert,
                "n_pairs": n,
                "question_text_changed_rate": sub["question_text_changed"].mean(),
                "question_length_diff_mean_words": qlen_diff.mean(),
                "question_length_diff_median_words": qlen_diff.median(),
                "question_length_diff_std_words": qlen_diff.std(),
                "options_length_diff_mean_words": opt_diff.mean() if len(opt_diff) else np.nan,
                "options_length_diff_median_words": opt_diff.median() if len(opt_diff) else np.nan,
                "lexical_similarity_mean": lex_sim.mean(),
                "lexical_similarity_median": lex_sim.median(),
                "lexical_similarity_std": lex_sim.std(),
                "answer_position_comparable": pos_applicable,
                "answer_position_change_rate": pos_change_rate,
                "raw_answer_token_change_rate": raw_token_change_rate,
                "is_task_inverted": bool(sub["is_task_inverted"].any()),
                "answer_format_distribution": json.dumps(fmt_dist),
            }
        )
    return pd.DataFrame(rows)


def fig_perturbation_counts(df: pd.DataFrame) -> None:
    counts = df.groupby(["perturbation_type", "source_dataset"]).size().unstack(fill_value=0)
    counts = counts.reindex(PERTURBATIONS)
    counts = counts[SOURCES]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(PERTURBATIONS))
    width = 0.25
    for i, source in enumerate(SOURCES):
        ax.bar(
            x + (i - 1) * width,
            counts[source].values,
            width=width,
            label=source,
            color=CAT_PALETTE[i],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(PERTURBATIONS, rotation=20, ha="right")
    ax.set_xlabel("Perturbation type")
    ax.set_ylabel("Number of pairing rows")
    ax.set_title("ReMedQA pairing row counts by perturbation type and source")
    ax.legend(title="Source", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_counts.png", dpi=200)
    plt.close(fig)


def fig_length_change(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    data = [df.loc[df["perturbation_type"] == p, "question_length_diff"].values for p in PERTURBATIONS]
    bp = ax.boxplot(
        data,
        tick_labels=PERTURBATIONS,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0, "color": INK_SECONDARY},
        capprops={"linewidth": 1.0, "color": INK_SECONDARY},
    )
    for patch, color in zip(bp["boxes"], CAT_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    ax.set_xticklabels(PERTURBATIONS, rotation=20, ha="right")
    ax.set_xlabel("Perturbation type")
    ax.set_ylabel("Question word-count change (perturbed - original)")
    ax.set_title("Question length change by perturbation type (outliers hidden)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "length_change_by_perturbation.png", dpi=200)
    plt.close(fig)


def fig_lexical_similarity(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    data = [df.loc[df["perturbation_type"] == p, "lexical_similarity"].values for p in PERTURBATIONS]
    bp = ax.boxplot(
        data,
        tick_labels=PERTURBATIONS,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0, "color": INK_SECONDARY},
        capprops={"linewidth": 1.0, "color": INK_SECONDARY},
    )
    for patch, color in zip(bp["boxes"], CAT_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticklabels(PERTURBATIONS, rotation=20, ha="right")
    ax.set_xlabel("Perturbation type")
    ax.set_ylabel("Lexical similarity (difflib ratio, 0-1)")
    ax.set_title(
        "Question-text lexical similarity by perturbation type\n"
        "(descriptive only - NOT a semantic-equivalence measure)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lexical_similarity_by_perturbation.png", dpi=200)
    plt.close(fig)


def fig_answer_position(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(POSITION_COMPARABLE_PERTURBATIONS), figsize=(16, 5.5), sharey=True)
    for ax, pert, color in zip(axes, POSITION_COMPARABLE_PERTURBATIONS, CAT_PALETTE[:3]):
        sub = df[df["perturbation_type"] == pert]
        max_pos = int(pd.concat([sub["orig_answer_position"], sub["pert_answer_position"]]).max())
        labels = LETTER_LABELS[: max_pos + 1]
        orig_counts = sub["orig_answer_position"].value_counts().reindex(range(max_pos + 1), fill_value=0)
        pert_counts = sub["pert_answer_position"].value_counts().reindex(range(max_pos + 1), fill_value=0)

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, orig_counts.values, width=width, label="original (mcq)", color=INK_SECONDARY, alpha=0.6)
        ax.bar(x + width / 2, pert_counts.values, width=width, label=f"perturbed ({pert})", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Correct-answer position")
        ax.set_title(pert)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Count of pairs")
    fig.suptitle(
        "Correct-answer position distribution: original vs. perturbed (position-index based)",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "answer_position_distribution.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PAIRS_PARQUET)
    print(f"[info] loaded {len(df)} pairing rows from {PAIRS_PARQUET}")

    df = enrich(df)

    desc = build_descriptives_table(df)
    desc.to_csv(DESCRIPTIVES_CSV, index=False)
    print(f"[info] wrote {DESCRIPTIVES_CSV} ({len(desc)} rows)")
    print(desc.to_string(index=False))

    fig_perturbation_counts(df)
    fig_length_change(df)
    fig_lexical_similarity(df)
    fig_answer_position(df)

    for fname in [
        "perturbation_counts.png",
        "length_change_by_perturbation.png",
        "lexical_similarity_by_perturbation.png",
        "answer_position_distribution.png",
    ]:
        fpath = FIGURES_DIR / fname
        assert fpath.exists(), f"Expected figure not created: {fpath}"
        print(f"[info] wrote {fpath}")


if __name__ == "__main__":
    main()
