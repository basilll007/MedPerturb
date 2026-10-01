"""Tables and metrics. Deliberately no single global robustness score:
every metric is reported per perturbation, and paired-consistency metrics
(ReAcc / ReCon) only where the registry says 'same answer' is meaningful."""

import json
import math

import pandas as pd

from medperturb.evaluation.semantic import evaluate, norm_text
from medperturb.parsing.canonical import parse
from medperturb.perturbations.registry import REGISTRY, CONDITIONS, NOTA_TEXT, output_spec_for

MEANING_PRESERVING = [c for c in CONDITIONS if c != "mcq" and REGISTRY[c].recon_defined]


def _thinking_tokens(resp):
    """Providers may omit the thinking count when none occurred. Only derive 0
    when the token arithmetic proves it (total == prompt + candidates)."""
    if resp.thinking_tokens is not None:
        return resp.thinking_tokens, "reported"
    u = (resp.raw_payload or {}).get("usage_metadata") or {}
    tot, pr, ca = u.get("total_token_count"), u.get("prompt_token_count"), u.get("candidates_token_count")
    if None not in (tot, pr, ca) and tot == pr + ca:
        return 0, "derived_zero_from_totals"
    return None, "unknown"


def build_responses(records) -> pd.DataFrame:
    rows = []
    for ex, prompt, resp, cached in records:
        think, think_src = _thinking_tokens(resp)
        spec = output_spec_for(ex)
        pr = parse(resp.raw_response, spec)
        ev = evaluate(ex, pr)
        truncated = (resp.finish_reason or "").upper() == "MAX_TOKENS"
        rows.append({
            "question_id": ex.question_id, "source_dataset": ex.source_dataset, "condition": ex.condition,
            "category": REGISTRY[ex.condition].category,
            "provider": resp.provider, "model": resp.model, "model_version": resp.model_version,
            "prompt": prompt, "raw_response": resp.raw_response,
            "raw_answer": json.dumps(pr.raw_answer) if pr.raw_answer is not None else None,
            "normalized_answer": json.dumps(pr.normalized_answer) if pr.normalized_answer is not None else None,
            "parse_success": pr.parse_success, "parse_failure_reason": pr.parse_failure_reason,
            "evaluable": ev.evaluable, "correct": ev.correct, "match_method": ev.match_method, "eval_note": ev.note,
            "selected_option_index": ev.selected_option_index, "selected_option_text": ev.selected_option_text,
            "selected_option_indices": json.dumps(ev.selected_option_indices) if ev.selected_option_indices is not None else None,
            "implied_option_index": ev.implied_option_index,
            "gold_label": json.dumps(ex.gold_label), "gold_option_index": ex.gold_index,
            "gold_option_text": ex.gold_text, "gold_option_indices": json.dumps(ex.gold_indices),
            "api_failure": resp.error is not None, "api_error": resp.error,
            "finish_reason": resp.finish_reason, "truncated": truncated,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "thinking_tokens": think, "thinking_tokens_source": think_src, "latency_s": resp.latency_s,
            "cost_usd": resp.cost_usd, "timestamp": resp.timestamp, "from_cache": cached,
        })
    return pd.DataFrame(rows)


def _answer_text(r):
    """Semantic identity of a response: the option text it resolves to (for
    'incorrect', the implied single option it left unmarked)."""
    if r["condition"] == "incorrect":
        return r["selected_option_text"]  # evaluator stores implied option's text here
    return r["selected_option_text"]


def _answer_position(r):
    if r["condition"] == "incorrect":
        return r["implied_option_index"]
    if r["condition"] == "open":
        return None  # no displayed positions
    return r["selected_option_index"]


def build_paired(resp: pd.DataFrame) -> pd.DataFrame:
    by_q = {q: g.set_index("condition", drop=False) for q, g in resp.groupby("question_id")}
    rows = []
    for q, g in by_q.items():
        m = g.loc["mcq"]
        for cond in CONDITIONS:
            if cond == "mcq":
                continue
            p = g.loc[cond]
            both = bool(m["evaluable"]) and bool(p["evaluable"])
            m_pos, p_pos = _answer_position(m), _answer_position(p)
            m_txt, p_txt = _answer_text(m), _answer_text(p)
            pos_changed = (m_pos != p_pos) if (both and m_pos is not None and p_pos is not None
                                                and not _nan(m_pos) and not _nan(p_pos)) else None
            sem_changed = (norm_text(m_txt) != norm_text(p_txt)) if (both and m_txt and p_txt and not _nan(m_txt) and not _nan(p_txt)) else None
            rows.append({
                "question_id": q, "source_dataset": m["source_dataset"], "condition": cond,
                "transition_type": REGISTRY[cond].transition_type,
                "mcq_evaluable": bool(m["evaluable"]), "perturbed_evaluable": bool(p["evaluable"]),
                "both_evaluable": both,
                "mcq_correct": m["correct"], "perturbed_correct": p["correct"],
                "mcq_selected_index": m_pos, "perturbed_selected_index": p_pos,
                "mcq_selected_text": m_txt, "perturbed_selected_text": p_txt,
                "mcq_gold_index": m["gold_option_index"], "perturbed_gold_index": p["gold_option_index"],
                "mcq_gold_text": m["gold_option_text"], "perturbed_gold_text": p["gold_option_text"],
                "position_changed": pos_changed, "semantic_answer_changed": sem_changed,
                "transition": _transition(m["correct"], p["correct"]) if both else None,
                "adaptation": _adaptation(cond, m, p) if both else None,
            })
    return pd.DataFrame(rows)


def _nan(x):
    return isinstance(x, float) and math.isnan(x)


def _transition(mc, pc):
    return f"{'C' if mc else 'W'}->{'C' if pc else 'W'}"


def _adaptation(cond, m, p):
    t = REGISTRY[cond].transition_type
    if t == "adaptation":  # none_of_the_provided: correct content removed -> must choose NOTA
        if p["correct"]:
            return "successful_adaptation"
        if norm_text(p["selected_option_text"]) == norm_text(m["selected_option_text"]):
            return "failure_to_adapt_kept_same_distractor"
        return "failure_to_adapt_chose_distractor"
    if t == "inversion":  # incorrect: must output the complement set
        if p["correct"]:
            return "successful_adaptation"
        idxs = json.loads(p["selected_option_indices"]) if p["selected_option_indices"] else []
        if len(idxs) != len(json.loads(p["gold_option_indices"])):
            return "failure_to_adapt_wrong_set_size"
        return "failure_to_adapt_included_gold"
    return None


def classify_failure(r, mcq_row) -> str | None:
    if r["api_failure"]:
        return "provider_api_failure"
    if r["truncated"]:
        return "max_tokens_truncation"
    if not r["parse_success"]:
        return "parse_failure"
    if not r["evaluable"]:
        return "evaluation_ambiguity"
    if r["condition"] == "no_symbols" and r["match_method"] in ("no_match",) or (
            r["condition"] == "no_symbols" and str(r["match_method"]).startswith("ambiguous")):
        return "invalid_option"
    if r["condition"] == "incorrect":
        idxs = json.loads(r["selected_option_indices"]) if r["selected_option_indices"] else []
        if len(idxs) != len(json.loads(r["gold_option_indices"])):
            return "instruction_following_failure"
    if r["correct"]:
        return None
    if r["condition"] == "mcq":
        return "knowledge_error"
    if r["condition"] == "none_of_the_provided":
        return "required_adaptation_failure"
    if mcq_row is None or not mcq_row["evaluable"]:
        return "unattributed_error"
    return "representation_instability" if mcq_row["correct"] else "knowledge_error"


def build_failures(resp: pd.DataFrame) -> pd.DataFrame:
    mcq = {r["question_id"]: r for _, r in resp[resp["condition"] == "mcq"].iterrows()}
    out = []
    for _, r in resp.iterrows():
        f = classify_failure(r, mcq.get(r["question_id"]))
        if f is None:
            continue
        out.append({"question_id": r["question_id"], "condition": r["condition"], "failure_type": f,
                    "finish_reason": r["finish_reason"], "parse_failure_reason": r["parse_failure_reason"],
                    "raw_response": r["raw_response"], "selected_option_text": r["selected_option_text"],
                    "gold_option_text": r["gold_option_text"], "match_method": r["match_method"],
                    "eval_note": r["eval_note"]})
    return pd.DataFrame(out, columns=["question_id", "condition", "failure_type", "finish_reason",
                                       "parse_failure_reason", "raw_response", "selected_option_text",
                                       "gold_option_text", "match_method", "eval_note"])


def _rate(num, den):
    return num / den if den else float("nan")


def metrics_by_perturbation(resp: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        spec = REGISTRY[cond]
        g = resp[resp["condition"] == cond]
        n = len(g)
        n_parsed = int(g["parse_success"].sum())
        n_eval = int(g["evaluable"].sum())
        n_correct = int((g["correct"] == True).sum())  # noqa: E712 - explicit, never pandas ~
        n_ambig = n_parsed - n_eval
        row = {
            "condition": cond, "category": spec.category, "N": n,
            "n_api_failures": int(g["api_failure"].sum()), "n_truncated": int(g["truncated"].sum()),
            "n_parsed": n_parsed, "parse_success_rate": _rate(n_parsed, n),
            "n_evaluable": n_eval, "n_evaluation_ambiguous": n_ambig, "n_correct": n_correct,
            "accuracy": _rate(n_correct, n_eval),
            "accuracy_lower_bound": _rate(n_correct, n_parsed) if cond == "open" else float("nan"),
            "accuracy_upper_bound": _rate(n_correct + n_ambig, n_parsed) if cond == "open" else float("nan"),
            "mean_input_tokens": g["input_tokens"].mean(), "mean_output_tokens": g["output_tokens"].mean(),
            "mean_thinking_tokens": g["thinking_tokens"].mean(), "max_thinking_tokens": g["thinking_tokens"].max(),
            "mean_latency_s": g["latency_s"].mean(), "cost_usd": g["cost_usd"].sum(),
        }
        if cond != "mcq":
            p = paired[(paired["condition"] == cond) & (paired["both_evaluable"] == True)]  # noqa: E712
            k = len(p)
            row.update({
                "n_pairs": k,
                **{f"n_{t.replace('->', '_to_')}": int((p["transition"] == t).sum()) for t in ("C->C", "C->W", "W->C", "W->W")},
                "position_change_rate": _rate(int((p["position_changed"] == True).sum()), int(p["position_changed"].notna().sum())),
                "semantic_answer_change_rate": _rate(int((p["semantic_answer_changed"] == True).sum()), int(p["semantic_answer_changed"].notna().sum())),
                "ReAcc": _rate(int(((p["mcq_correct"] == True) & (p["perturbed_correct"] == True)).sum()), k) if spec.recon_defined else float("nan"),
                "ReCon": _rate(int((p["semantic_answer_changed"] == False).sum()), int(p["semantic_answer_changed"].notna().sum())) if spec.recon_defined else float("nan"),
                "recon_defined": spec.recon_defined,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def group_consistency(resp: pd.DataFrame) -> dict:
    """ReAcc/ReCon across ALL meaning-preserving representations jointly
    (mcq + roman_numeral + fixed_pos + no_symbols), per ReMedQA's definitions:
    ReAcc = correct on every version; ReCon = same answer on every version."""
    conds = ["mcq"] + MEANING_PRESERVING
    reacc_n = recon_n = den = 0
    for q, g in resp.groupby("question_id"):
        g = g.set_index("condition", drop=False)
        if not all(bool(g.loc[c, "evaluable"]) for c in conds):
            continue
        den += 1
        if all(g.loc[c, "correct"] is True or g.loc[c, "correct"] == True for c in conds):  # noqa: E712
            reacc_n += 1
        texts = {norm_text(g.loc[c, "selected_option_text"]) for c in conds}
        if len(texts) == 1:
            recon_n += 1
    return {"conditions": conds, "n_questions_all_evaluable": den,
            "ReAcc_all": _rate(reacc_n, den), "ReCon_all": _rate(recon_n, den)}
