"""Phase 2A figures, per perturbation - deliberately no aggregate robustness plot.

Run after Stage 3: uv run python scripts\\phase2a_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "behavioral" / "phase2a"
OUT = ROOT / "figures" / "behavioral" / "phase2a"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # validated categorical slots 1-4
ORDER = ["mcq", "roman_numeral", "fixed_pos", "no_symbols", "none_of_the_provided", "incorrect", "open"]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "axes.titlesize": 13, "axes.titleweight": "semibold",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    m = pd.read_csv(IN / "metrics_by_perturbation.csv").set_index("condition").loc[ORDER]
    n = int(m["N"].iloc[0])
    labels = list(reversed(ORDER))

    # 1. Accuracy where defined (open: bounds only, never a point estimate)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, c in enumerate(labels):
        r = m.loc[c]
        if c == "open":
            lo, hi = r["accuracy_lower_bound"], r["accuracy_upper_bound"]
            ax.plot([lo * 100, hi * 100], [i, i], color=INK2, linewidth=2, solid_capstyle="round")
            ax.text(lo * 100, i + 0.32, f"undefined — only {int(r['n_evaluable'])}/{int(r['n_parsed'])} evaluable; "
                    f"bounds {lo:.0%}–{hi:.0%}", va="bottom", color=INK2, fontsize=9)
        else:
            ax.barh(i, r["accuracy"] * 100, height=0.55, color=BLUE)
            ax.text(r["accuracy"] * 100 + 1.5, i, f"{r['accuracy']:.0%}  ({int(r['n_correct'])}/{int(r['n_evaluable'])})",
                    va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Accuracy among evaluable responses (%)")
    ax.set_title(f"Accuracy by perturbation\ngemini-3.8-flash · N={n} per condition · framework validation", loc="left")
    _style(ax)
    _save(fig, "accuracy_by_perturbation.png")

    # 2. Parse-success rate
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, c in enumerate(labels):
        r = m.loc[c]
        ax.barh(i, r["parse_success_rate"] * 100, height=0.55, color=BLUE)
        ax.text(r["parse_success_rate"] * 100 + 1.5, i,
                f"{r['parse_success_rate']:.0%}  (truncated: {int(r['n_truncated'])})", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 125)
    ax.set_xlabel("Parse-success rate (%)")
    ax.set_title(f"Parse success by perturbation\nN={n} per condition", loc="left")
    _style(ax)
    _save(fig, "parse_success_by_perturbation.png")

    # 3. Correctness transitions vs mcq (open excluded: not evaluable)
    t = pd.read_csv(IN / "transitions.csv")
    t = t[t["kind"] == "correctness_transition"]
    conds = [c for c in ORDER if c not in ("mcq", "open")]
    cats = [("C->C", BLUE), ("C->W", ORANGE), ("W->C", AQUA), ("W->W", YELLOW)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    yl = list(reversed(conds))
    for i, c in enumerate(yl):
        left = 0
        for lab, col in cats:
            k = int(t[(t["condition"] == c) & (t["label"] == lab)]["count"].sum())
            if k:
                ax.barh(i, k, left=left, height=0.55, color=col, edgecolor=SURFACE, linewidth=2)
                if k >= 2:
                    ax.text(left + k / 2, i, str(k), ha="center", va="center", color=INK, fontsize=9.5)
                left += k
        small = [f"{lab.replace('->', '→')} {int(t[(t['condition'] == c) & (t['label'] == lab)]['count'].sum())}"
                 for lab, _ in cats if 0 < int(t[(t["condition"] == c) & (t["label"] == lab)]["count"].sum()) < 2]
        if small:
            ax.text(left + 0.6, i, " · ".join(small), va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(yl)), yl)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Paired questions (both conditions evaluable)")
    ax.set_title("Correctness transitions, mcq → perturbation\nopen excluded (not evaluable); counts < 2 labeled at right", loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, col in cats],
              labels=["correct → correct", "correct → wrong", "wrong → correct", "wrong → wrong"],
              ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9.5)
    _style(ax)
    _save(fig, "correctness_transitions.png")

    # 4. Token usage (billed output = visible + thinking; same unit, stacked)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, c in enumerate(labels):
        r = m.loc[c]
        vis, th = r["mean_output_tokens"], r["mean_thinking_tokens"]
        ax.barh(i, th, height=0.55, color=ORANGE, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, vis, left=th, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.text(th + vis + 4, i, f"{th:.0f} thinking + {vis:.0f} visible  (max thinking {int(r['max_thinking_tokens'])})",
                va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, max((m["mean_thinking_tokens"] + m["mean_output_tokens"]).max() * 2.1, 100))
    ax.set_xlabel("Mean tokens per response (thinking budget 1024; shared ceiling 1536)")
    ax.set_title("Token usage by perturbation\nbilled output = thinking + visible tokens", loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["thinking tokens", "visible output tokens"], frameon=False, loc="lower right", fontsize=9.5)
    _style(ax)
    _save(fig, "token_usage_by_perturbation.png")
    print(f"Wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
