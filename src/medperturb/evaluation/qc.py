"""Quality control: structural assertions + independent plain-Python
recomputation of key metrics (no pandas boolean ops, no shared code path
with metrics.py). Any disagreement is a STOP condition."""

import json
import math

from medperturb.perturbations.registry import REGISTRY, CONDITIONS


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def _true(x):
    """Strict truth: None/NaN are not True (bool(NaN) would be); numpy.bool_ handled."""
    return x is not None and not _is_nan(x) and bool(x) is True


def _false(x):
    return x is not None and not _is_nan(x) and bool(x) is False


def _eq(a, b, tol=1e-12):
    if (a is None or (isinstance(a, float) and math.isnan(a))) and (b is None or (isinstance(b, float) and math.isnan(b))):
        return True
    return a is not None and b is not None and abs(a - b) <= tol


def structural_checks(examples, records, responses_df, n_questions, config_hash_expected) -> list[str]:
    errs = []
    keys = [(ex.question_id, ex.condition) for ex in examples]
    if len(keys) != n_questions * len(CONDITIONS):
        errs.append(f"expected {n_questions * len(CONDITIONS)} examples, got {len(keys)}")
    if len(set(keys)) != len(keys):
        errs.append("duplicate (question, condition) examples")
    rec_keys = [(ex.question_id, ex.condition) for ex, _, _, _ in records]
    if sorted(rec_keys) != sorted(keys):
        errs.append("records do not cover exactly the requested examples (silent missing or extra results)")
    if len(set(rec_keys)) != len(rec_keys):
        errs.append("duplicate result records")
    qs = {}
    for q, c in keys:
        qs.setdefault(q, set()).add(c)
    for q, cs in qs.items():
        if cs != set(CONDITIONS):
            errs.append(f"{q}: incomplete condition set {sorted(cs)}")
    for ex in examples:
        n = len(ex.option_texts)
        if n != 4:
            errs.append(f"{ex.question_id}/{ex.condition}: option count {n}")
        if ex.condition == "incorrect":
            if not ex.gold_indices or any(i < 0 or i >= n for i in ex.gold_indices):
                errs.append(f"{ex.question_id}/incorrect: invalid gold set")
        elif ex.gold_index is None or not (0 <= ex.gold_index < n):
            errs.append(f"{ex.question_id}/{ex.condition}: invalid gold index {ex.gold_index}")
    if len(responses_df) != len(keys):
        errs.append(f"responses table has {len(responses_df)} rows, expected {len(keys)}")
    return errs


def independent_recompute(responses_df, metrics_df, paired_df, group) -> list[str]:
    rows = responses_df.to_dict("records")
    errs = []
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question_id"], {})[r["condition"]] = r

    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        n = len(rs)
        parsed = sum(1 for r in rs if _true(r["parse_success"]))
        evaluable = sum(1 for r in rs if _true(r["evaluable"]))
        correct = sum(1 for r in rs if _true(r["correct"]))
        incorrect = sum(1 for r in rs if _false(r["correct"]))
        if evaluable != correct + incorrect:
            errs.append(f"{cond}: evaluable {evaluable} != correct {correct} + incorrect {incorrect}")
        if evaluable > parsed or parsed > n:
            errs.append(f"{cond}: denominators out of order (N={n}, parsed={parsed}, evaluable={evaluable})")
        m = metrics_df[metrics_df["condition"] == cond].iloc[0].to_dict()
        for name, mine in (("N", n), ("n_parsed", parsed), ("n_evaluable", evaluable), ("n_correct", correct),
                           ("parse_success_rate", parsed / n if n else float("nan")),
                           ("accuracy", correct / evaluable if evaluable else float("nan"))):
            if not _eq(float(mine), float(m[name])):
                errs.append(f"{cond}.{name}: independent={mine} reported={m[name]}")

        if cond == "mcq":
            continue
        cc = cw = wc = ww = pairs = same = semden = 0
        for q, conds in by_q.items():
            a, b = conds["mcq"], conds[cond]
            if not (_true(a["evaluable"]) and _true(b["evaluable"])):
                continue
            pairs += 1
            key = ("C" if _true(a["correct"]) else "W") + ("C" if _true(b["correct"]) else "W")
            cc += key == "CC"; cw += key == "CW"; wc += key == "WC"; ww += key == "WW"
            ta, tb = a["selected_option_text"], b["selected_option_text"]
            if ta and tb and ta == ta and tb == tb:  # exclude NaN
                semden += 1
                from medperturb.evaluation.semantic import norm_text
                same += norm_text(ta) == norm_text(tb)
        for name, mine in (("n_pairs", pairs), ("n_C_to_C", cc), ("n_C_to_W", cw), ("n_W_to_C", wc), ("n_W_to_W", ww)):
            if int(m[name]) != mine:
                errs.append(f"{cond}.{name}: independent={mine} reported={m[name]}")
        if REGISTRY[cond].recon_defined:
            if not _eq(cc / pairs if pairs else float("nan"), float(m["ReAcc"])):
                errs.append(f"{cond}.ReAcc: independent={cc / pairs if pairs else None} reported={m['ReAcc']}")
            if not _eq(same / semden if semden else float("nan"), float(m["ReCon"])):
                errs.append(f"{cond}.ReCon: independent={same / semden if semden else None} reported={m['ReCon']}")

    # joint ReAcc/ReCon across meaning-preserving set
    conds = group["conditions"]
    den = reacc = recon = 0
    from medperturb.evaluation.semantic import norm_text
    for q, cs in by_q.items():
        if not all(_true(cs[c]["evaluable"]) for c in conds):
            continue
        den += 1
        reacc += all(_true(cs[c]["correct"]) for c in conds)
        recon += len({norm_text(cs[c]["selected_option_text"]) for c in conds}) == 1
    if den != group["n_questions_all_evaluable"]:
        errs.append(f"group denominator: independent={den} reported={group['n_questions_all_evaluable']}")
    if not _eq(reacc / den if den else float("nan"), group["ReAcc_all"]):
        errs.append(f"ReAcc_all: independent={reacc / den if den else None} reported={group['ReAcc_all']}")
    if not _eq(recon / den if den else float("nan"), group["ReCon_all"]):
        errs.append(f"ReCon_all: independent={recon / den if den else None} reported={group['ReCon_all']}")
    return errs
