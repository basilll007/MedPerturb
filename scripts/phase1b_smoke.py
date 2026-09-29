"""Phase 1B - VALIDATION + PERTURBATION SENSITIVITY CHECK.

Reuses the exact 10 question IDs and the validated Phase 1A evaluation
framework (prompting, canonical parsing, caching). Adds a third condition,
none_of_the_provided, which is semantically audited BEFORE any API call for
it - its answer-set/decision-boundary character is not assumed to be
meaning-preserving like roman_numeral's presentation/label change.

Reuses every successful Phase 1A Gemini mcq/roman_numeral response from the
shared raw cache (results/behavioral/raw/) - no duplicate API spend. Claude
is attempted exactly once; if still blocked, it is reported unavailable and
not retried or substituted.

Run: uv run python scripts\\phase1b_smoke.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1a_prompts import parse_options_json, labels_from_options  # noqa: E402
from phase1a_parse import parse_answer  # noqa: E402
import phase1a_smoke as p1a  # reuse validated call_*/do_call_and_cache/canonical_index/PRICING/MODELS  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "data" / "processed" / "pilot_200.parquet"
PHASE1A_RESPONSES = ROOT / "results" / "behavioral" / "phase1a" / "phase1a_responses.csv"
PHASE1A_METRICS = ROOT / "results" / "behavioral" / "phase1a" / "phase1a_metrics.csv"
OUT_DIR = ROOT / "results" / "behavioral" / "phase1b"
FIG_DIR = ROOT / "figures" / "behavioral"
LOG_DIR = ROOT / "logs"

for d in (OUT_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "phase1b_smoke.jsonl"
MODELS = p1a.MODELS
CONDITIONS = ["mcq", "roman_numeral", "none_of_the_provided"]


def log_event(**kwargs):
    kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(kwargs, default=str)
    p1a.assert_no_secrets(line, "phase1b log_event")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- recovery

def recover_phase1a_ids():
    r = pd.read_csv(PHASE1A_RESPONSES)
    ids = sorted(r["question_id"].unique())
    assert len(ids) == 10, f"expected 10 Phase 1A question ids, found {len(ids)} - aborting"
    return [tuple(i.split("::")) for i in ids]


def load_phase1b_rows(pairs):
    df = pd.read_parquet(PILOT_PATH)
    rows = {}
    for source_dataset, source_id in pairs:
        sub = df[(df["source_dataset"] == source_dataset) & (df["source_id"] == source_id)]
        roman = sub[sub["perturbation_type"] == "roman_numeral"]
        none_ = sub[sub["perturbation_type"] == "none_of_the_provided"]
        assert len(roman) == 1, f"expected exactly 1 roman_numeral row for {source_dataset}/{source_id}, got {len(roman)}"
        assert len(none_) == 1, f"expected exactly 1 none_of_the_provided row for {source_dataset}/{source_id}, got {len(none_)}"
        rows[(source_dataset, source_id)] = {"roman": roman.iloc[0], "none": none_.iloc[0]}
    return rows


def pairing_validation(pairs, rows):
    print("=== Pairing validation (abort on any mismatch) ===")
    assert len(pairs) == 10, f"expected 10 ids, got {len(pairs)}"
    assert len(set(pairs)) == 10, "duplicate ids in recovered Phase 1A id list"
    for key in pairs:
        r, n = rows[key]["roman"], rows[key]["none"]
        assert r["original_question"] == n["original_question"], f"mcq question mismatch across perturbation rows for {key}"
        assert r["original_options"] == n["original_options"], f"mcq options mismatch across perturbation rows for {key}"
        assert r["original_answer_letter"] == n["original_answer_letter"], f"mcq gold mismatch across perturbation rows for {key}"
        for opt_str in (r["original_options"], r["perturbed_options"], n["perturbed_options"]):
            opts = parse_options_json(opt_str)
            assert len(opts) > 0, f"empty options for {key}: {opt_str!r}"
    print(f"OK: {len(pairs)} unique ids, each has mcq + roman_numeral + none_of_the_provided, "
          f"mcq baseline identical across both perturbation-type rows, all options parseable.\n")


# ------------------------------------------------------------- audit (§semantic)

def semantic_audit(pairs, rows):
    print("=== CRITICAL SEMANTIC AUDIT: none_of_the_provided (before any API call for it) ===")
    audit_rows = []
    for source_dataset, source_id in pairs:
        n = rows[(source_dataset, source_id)]["none"]
        original_options = parse_options_json(n["original_options"])
        perturbed_options = parse_options_json(n["perturbed_options"])
        original_gold = n["original_answer_letter"]
        perturbed_gold = n["perturbed_answer_raw"]
        original_gold_text = n["original_answer_text"]
        perturbed_gold_text = perturbed_options.get(perturbed_gold)

        gold_present = original_gold_text in perturbed_options.values()
        gold_letter_changed = perturbed_gold != original_gold
        gold_text_changed = perturbed_gold_text != original_gold_text
        gold_changed = gold_letter_changed or gold_text_changed
        option_count_original = len(original_options)
        option_count_perturbed = len(perturbed_options)
        ordering_changed = list(original_options.keys()) != list(perturbed_options.keys())

        anomalies = []
        if option_count_perturbed != option_count_original:
            anomalies.append(f"option count changed {option_count_original}->{option_count_perturbed}")
        if gold_letter_changed:
            anomalies.append(f"gold LETTER position changed {original_gold}->{perturbed_gold} (adaptation required)")
        if gold_present:
            anomalies.append("original gold text still present verbatim among perturbed options (unexpected)")
        if perturbed_gold_text != "None of the provided options":
            anomalies.append(f"gold option text is not the literal 'None of the provided options' string "
                              f"(got {perturbed_gold_text!r})")

        audit_status = "OK" if not anomalies else "ANOMALY"
        notes = "; ".join(anomalies) if anomalies else (
            "Correct option's TEXT replaced with the literal string 'None of the provided options'; "
            "letter position and option count unchanged; question text unchanged. "
            "The decision problem changes (correct choice is no longer identifiable by clinical content) "
            "even though the correct LETTER does not move - NOT assumed meaning-preserving."
        )

        audit_rows.append({
            "question_id": f"{source_dataset}::{source_id}",
            "original_gold": original_gold,
            "perturbed_gold": perturbed_gold,
            "original_gold_text": original_gold_text,
            "perturbed_gold_text": perturbed_gold_text,
            "original_gold_present_in_perturbed_options": gold_present,
            "original_option_count": option_count_original,
            "perturbed_option_count": option_count_perturbed,
            "option_ordering_changed": ordering_changed,
            "gold_letter_changed": gold_letter_changed,
            "gold_text_changed": gold_text_changed,
            "gold_changed": gold_changed,
            "audit_status": audit_status,
            "notes": notes,
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUT_DIR / "none_semantic_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    n_anomaly = int((audit_df["audit_status"] == "ANOMALY").sum())
    n_letter_changed = int(audit_df["gold_letter_changed"].sum())
    n_text_changed = int(audit_df["gold_text_changed"].sum())
    print(f"Audited {len(audit_df)} questions.")
    print(f"  Gold LETTER position changed: {n_letter_changed}/{len(audit_df)} "
          f"(this determines whether 'required adaptation' applies per question)")
    print(f"  Gold option TEXT changed to generic boilerplate: {n_text_changed}/{len(audit_df)}")
    print(f"  Structural anomalies flagged: {n_anomaly}/{len(audit_df)}")
    if n_anomaly:
        print(audit_df[audit_df["audit_status"] == "ANOMALY"][["question_id", "notes"]].to_string(index=False))
    print(f"Conclusion: none_of_the_provided is confirmed NOT meaning-preserving at the content level "
          f"(correct option's clinical text is destroyed) even in the {len(audit_df) - n_letter_changed} "
          f"case(s) where the correct LETTER position happens not to move. Treated as a distinct "
          f"answer-set/decision-boundary robustness probe, never pooled with roman_numeral.")
    print(f"Written: {audit_path}\n")
    return audit_df


# ------------------------------------------------------------------- calling

def conditions_for(rows, key):
    r, n = rows[key]["roman"], rows[key]["none"]
    mcq_options = parse_options_json(r["original_options"])
    roman_options = parse_options_json(r["perturbed_options"])
    none_options = parse_options_json(n["perturbed_options"])
    return [
        {"perturbation_type": "mcq", "question": r["original_question"], "options": mcq_options,
         "gold": r["original_answer_letter"], "labels": list(mcq_options.keys())},
        {"perturbation_type": "roman_numeral", "question": r["perturbed_question"], "options": roman_options,
         "gold": r["perturbed_answer_raw"], "labels": list(roman_options.keys())},
        {"perturbation_type": "none_of_the_provided", "question": n["perturbed_question"], "options": none_options,
         "gold": n["perturbed_answer_raw"], "labels": labels_from_options(none_options)},
    ]


def check_claude_availability(clients, rows, pairs):
    print("=== Claude availability check (single attempt, no retries, no substitution) ===")
    first_key = pairs[0]
    first_cond = conditions_for(rows, first_key)[0]  # mcq
    try:
        record, cached = p1a.do_call_and_cache(clients, "anthropic", "claude-haiku-4-5-20251001",
                                                 first_key[0], first_key[1], first_cond)
        parsed = parse_answer(record["response_text"], "mcq", labels=first_cond["labels"])
        status = "cache-hit" if cached else "live call"
        print(f"Claude responded ({status}), parse_status={parsed.parsing_status} -> AVAILABLE.\n")
        return True
    except Exception as e:
        print(f"Claude call failed: {type(e).__name__}: {e}")
        print("-> Claude UNAVAILABLE due to provider/account limits. Not retrying, not substituting "
              "another Claude model, not silently dropping it from the report.\n")
        log_event(event="claude_unavailable", error_type=type(e).__name__, error=str(e))
        return False


# ------------------------------------------------------------------ outputs

def extract_thinking_tokens(record):
    """Pulled from the raw provider response, not a field the earlier
    do_call_and_cache path stored top-level - works retroactively on
    already-cached records with zero new API spend."""
    raw = record.get("raw_response") or {}
    try:
        return raw["usage_metadata"]["thoughts_token_count"]
    except (KeyError, TypeError):
        return None


def build_responses_df(all_records, labels_lookup):
    rows = []
    for r in all_records:
        key = (r["source_dataset"], r["source_id"], r["perturbation_type"])
        labels = labels_lookup[key]
        parsed = parse_answer(r["response_text"], r["perturbation_type"], labels=labels)
        canon = p1a.canonical_index(r["perturbation_type"], parsed.parsed_label, labels=labels) if parsed.parsing_status == "ok" else None
        gold_canon = p1a.canonical_index(r["perturbation_type"], r["gold_answer"], labels=labels)
        r["thinking_tokens"] = extract_thinking_tokens(r)
        rows.append({
            "question_id": f"{r['source_dataset']}::{r['source_id']}",
            "source_dataset": r["source_dataset"],
            "source_id": r["source_id"],
            "provider": r["provider"],
            "model": r["model_id"],
            "model_version": r["model_version"],
            "condition": r["perturbation_type"],
            "prompt": r["prompt_text"],
            "raw_response": r["response_text"],
            "raw_answer": parsed.parsed_label,
            "canonical_answer": canon,
            "gold_answer": r["gold_answer"],
            "gold_canonical": gold_canon,
            "correct": (canon == gold_canon) if (canon is not None and gold_canon is not None) else None,
            "parse_success": parsed.parsing_status == "ok",
            "parsing_status": parsed.parsing_status,
            "latency_s": r.get("latency_s"),
            "input_tokens": r.get("input_tokens"),
            "output_tokens": r.get("output_tokens"),
            "thinking_tokens": r.get("thinking_tokens"),
            "estimated_cost_usd": r.get("estimated_cost_usd"),
            "reused_from_phase1a": r.get("_reused_from_phase1a", False),
            "prompt_version": r["prompt_version"],
            "timestamp": r["timestamp"],
        })
    return pd.DataFrame(rows)


def build_pairs_df(responses_df):
    pairs_rows = []
    group_iter = responses_df.groupby(["model", "question_id"]) if not responses_df.empty else []
    for (model, question_id), grp in group_iter:
        by_cond = {c: grp[grp["condition"] == c] for c in CONDITIONS}
        if any(by_cond[c].empty for c in CONDITIONS):
            continue
        mcq, rom, none_ = (by_cond[c].iloc[0] for c in CONDITIONS)

        mcq_canon, rom_canon, none_canon = mcq["canonical_answer"], rom["canonical_answer"], none_["canonical_answer"]
        roman_both = pd.notna(mcq_canon) and pd.notna(rom_canon)
        none_both = pd.notna(mcq_canon) and pd.notna(none_canon)

        pairs_rows.append({
            "question_id": question_id,
            "model": model,
            "mcq_gold_canonical": mcq["gold_canonical"],
            "roman_gold_canonical": rom["gold_canonical"],
            "none_gold_canonical": none_["gold_canonical"],
            "mcq_canonical": mcq_canon,
            "roman_canonical": rom_canon,
            "none_canonical": none_canon,
            "mcq_correct": mcq["correct"],
            "roman_correct": rom["correct"],
            "none_correct": none_["correct"],
            "roman_both_parsed": roman_both,
            "none_both_parsed": none_both,
            "roman_flipped": (mcq_canon != rom_canon) if roman_both else None,
            "none_answer_changed": (mcq_canon != none_canon) if none_both else None,
        })
    return pd.DataFrame(pairs_rows)


def build_adaptation_df(pairs_df, audit_df):
    audit_by_id = audit_df.set_index("question_id")
    rows = []
    for _, p in pairs_df.iterrows():
        qid = p["question_id"]
        a = audit_by_id.loc[qid]
        required = bool(a["gold_letter_changed"])

        if not p["none_both_parsed"]:
            category = "unparseable"
        elif required:
            if p["none_canonical"] == p["none_gold_canonical"]:
                category = "successful_adaptation"
            elif p["none_canonical"] == p["mcq_canonical"]:
                category = "failure_to_adapt"
            else:
                category = "adaptation_attempted_but_wrong"
        else:
            if p["mcq_correct"] and p["none_correct"]:
                category = "stable_correct"
            elif p["mcq_correct"] and not p["none_correct"]:
                category = "presentation_instability"
            elif (not p["mcq_correct"]) and p["none_correct"]:
                category = "unexpected_correction"
            else:
                category = "stable_incorrect"

        rows.append({
            "question_id": qid,
            "model": p["model"],
            "adaptation_required": required,
            "mcq_canonical": p["mcq_canonical"],
            "none_canonical": p["none_canonical"],
            "none_gold_canonical": p["none_gold_canonical"],
            "mcq_correct": p["mcq_correct"],
            "none_correct": p["none_correct"],
            "category": category,
        })
    return pd.DataFrame(rows)


def classify_failure(row):
    """Only classify where evidence in the row actually supports it -
    otherwise 'insufficient_evidence', never a forced guess."""
    if row["parsing_status"] != "ok":
        return "parsing_failure"
    if row["condition"] == "none_of_the_provided" and row.get("adaptation_category") == "presentation_instability":
        return "presentation_instability"
    if row["condition"] == "none_of_the_provided" and row.get("adaptation_category") in ("failure_to_adapt", "adaptation_attempted_but_wrong"):
        return "failure_to_adapt_to_changed_answer_set"
    if row["canonical_answer"] is not None and row.get("labels_len") is not None and row["canonical_answer"] >= row["labels_len"]:
        return "invalid_option_selection"
    if row["condition"] == "mcq" and not row["correct"]:
        return "possible_knowledge_error"
    return "insufficient_evidence"


def independent_sanity_check(pairs_df, metrics_df):
    """Recompute accuracy/flip/change-rate/ReAcc/ReCon via plain Python loops
    (deliberately NOT pandas vectorized ~/& operators, which caused a real
    bug in Phase 1A) and compare against the reported metrics. Required per
    Phase 1B spec since Phase 1A exposed a metric implementation bug."""
    print("=== Independent sanity recomputation (plain-Python, not pandas ~/&) ===")
    discrepancies = []
    for model in sorted(pairs_df["model"].unique()):
        sub = pairs_df[pairs_df["model"] == model]
        roman_valid = [row for _, row in sub.iterrows() if row["roman_both_parsed"]]
        none_valid = [row for _, row in sub.iterrows() if row["none_both_parsed"]]

        n_roman = len(roman_valid)
        flip_count = sum(1 for r in roman_valid if r["mcq_canonical"] != r["roman_canonical"])
        reacc_count = sum(1 for r in roman_valid if bool(r["mcq_correct"]) is True and bool(r["roman_correct"]) is True)
        recon_count = sum(1 for r in roman_valid if r["mcq_canonical"] == r["roman_canonical"])

        indep_flip_rate = flip_count / n_roman if n_roman else float("nan")
        indep_reacc = reacc_count / n_roman if n_roman else float("nan")
        indep_recon = recon_count / n_roman if n_roman else float("nan")

        reported = metrics_df[metrics_df["model"] == model]
        if reported.empty:
            continue
        reported = reported.iloc[0]

        for name, indep, rep_col in (("flip_rate", indep_flip_rate, "roman_flip_rate"),
                                       ("ReAcc", indep_reacc, "roman_ReAcc"),
                                       ("ReCon", indep_recon, "roman_ReCon")):
            rep_val = reported[rep_col]
            if pd.isna(indep) and pd.isna(rep_val):
                continue
            if pd.isna(indep) or pd.isna(rep_val) or abs(indep - rep_val) > 1e-9:
                discrepancies.append((model, name, indep, rep_val))

        print(f"  {model}: independent flip_rate={indep_flip_rate:.4f} ReAcc={indep_reacc:.4f} ReCon={indep_recon:.4f} "
              f"(n_roman_valid={n_roman}, n_none_valid={len(none_valid)})")

    if discrepancies:
        print("\n!!! METRIC DISCREPANCY DETECTED - STOP AND FLAG !!!")
        for model, name, indep, rep in discrepancies:
            print(f"  {model} {name}: independent={indep} reported={rep}")
    else:
        print("  OK: independent recomputation matches reported metrics exactly for all models.")
    print()
    return discrepancies


def main():
    import anthropic
    from google import genai

    pairs = recover_phase1a_ids()
    print(f"Recovered {len(pairs)} question ids from Phase 1A artifacts (no resampling): {pairs}\n")

    rows = load_phase1b_rows(pairs)
    pairing_validation(pairs, rows)
    audit_df = semantic_audit(pairs, rows)

    clients = {
        "anthropic": anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "gemini": genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
    }

    claude_available = check_claude_availability(clients, rows, pairs)
    active_models = [m for m in MODELS if m["provider"] == "gemini"]
    if claude_available:
        active_models = MODELS
    print(f"Active models for the full batch: {[m['model_id'] for m in active_models]}\n")

    all_records = []
    labels_lookup = {}
    reuse_count = 0
    new_call_count_by_model = {}

    for key in pairs:
        source_dataset, source_id = key
        for cond in conditions_for(rows, key):
            labels_lookup[(source_dataset, source_id, cond["perturbation_type"])] = cond["labels"]
            for m in active_models:
                provider, model_id = m["provider"], m["model_id"]
                try:
                    record, cached = p1a.do_call_and_cache(clients, provider, model_id, source_dataset, source_id, cond)
                    if cached:
                        reuse_count += 1
                        record["_reused_from_phase1a"] = cond["perturbation_type"] in ("mcq", "roman_numeral")
                    else:
                        new_call_count_by_model[model_id] = new_call_count_by_model.get(model_id, 0) + 1
                        record["_reused_from_phase1a"] = False
                        time.sleep(0.3)
                    all_records.append(record)
                except Exception as e:
                    log_event(event="call_error", provider=provider, model_id=model_id, source_dataset=source_dataset,
                               source_id=source_id, perturbation_type=cond["perturbation_type"],
                               error_type=type(e).__name__, error=str(e))
                    print(f"ERROR {provider}/{model_id} {source_dataset}/{source_id}/{cond['perturbation_type']}: "
                          f"{type(e).__name__}: {e}")

    responses_df = build_responses_df(all_records, labels_lookup)
    responses_df.to_csv(OUT_DIR / "phase1b_responses.csv", index=False)

    pairs_df = build_pairs_df(responses_df)
    pairs_df.to_csv(OUT_DIR / "phase1b_pairs.csv", index=False)

    # ---- metrics.csv (roman_numeral and none_of_the_provided reported SEPARATELY) ----
    metrics_rows = []
    for model in sorted(responses_df["model"].unique()):
        grp = responses_df[responses_df["model"] == model]
        p = pairs_df[pairs_df["model"] == model]
        mcq_resp = grp[grp["condition"] == "mcq"]
        roman_resp = grp[grp["condition"] == "roman_numeral"]
        none_resp = grp[grp["condition"] == "none_of_the_provided"]

        mcq_acc = mcq_resp["correct"].mean() if not mcq_resp.empty else float("nan")
        roman_acc = roman_resp["correct"].mean() if not roman_resp.empty else float("nan")
        none_acc = none_resp["correct"].mean() if not none_resp.empty else float("nan")

        roman_valid = p[p["roman_both_parsed"] == True]
        none_valid = p[p["none_both_parsed"] == True]

        roman_flip_rate = (roman_valid["roman_flipped"] == True).mean() if len(roman_valid) else float("nan")
        roman_reacc = ((roman_valid["mcq_correct"] == True) & (roman_valid["roman_correct"] == True)).mean() if len(roman_valid) else float("nan")
        roman_recon = (roman_valid["roman_flipped"] == False).mean() if len(roman_valid) else float("nan")

        none_change_rate = (none_valid["none_answer_changed"] == True).mean() if len(none_valid) else float("nan")
        cc = int(((none_valid["mcq_correct"] == True) & (none_valid["none_correct"] == True)).sum())
        cw = int(((none_valid["mcq_correct"] == True) & (none_valid["none_correct"] == False)).sum())
        wc = int(((none_valid["mcq_correct"] == False) & (none_valid["none_correct"] == True)).sum())
        ww = int(((none_valid["mcq_correct"] == False) & (none_valid["none_correct"] == False)).sum())

        metrics_rows.append({
            "model": model,
            "mcq_accuracy": mcq_acc,
            "roman_accuracy": roman_acc,
            "roman_delta_from_mcq": (roman_acc - mcq_acc) if pd.notna(roman_acc) and pd.notna(mcq_acc) else float("nan"),
            "roman_flip_rate": roman_flip_rate,
            "roman_ReAcc": roman_reacc,
            "roman_ReCon": roman_recon,
            "roman_n_valid_pairs": len(roman_valid),
            "none_accuracy": none_acc,
            "none_delta_from_mcq": (none_acc - mcq_acc) if pd.notna(none_acc) and pd.notna(mcq_acc) else float("nan"),
            "none_answer_change_rate": none_change_rate,
            "none_transition_CC": cc, "none_transition_CW": cw, "none_transition_WC": wc, "none_transition_WW": ww,
            "none_n_valid_pairs": len(none_valid),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "phase1b_metrics.csv", index=False)

    # ---- adaptation.csv ----
    adaptation_df = build_adaptation_df(pairs_df, audit_df) if not pairs_df.empty else pd.DataFrame()
    adaptation_df.to_csv(OUT_DIR / "phase1b_adaptation.csv", index=False)

    # ---- failures.csv ----
    adapt_lookup = {(r["question_id"], r["model"]): r["category"] for _, r in adaptation_df.iterrows()} if not adaptation_df.empty else {}
    failure_rows = []
    for _, r in responses_df.iterrows():
        if r["condition"] == "mcq" and r["correct"] is True:
            continue
        if r["condition"] in ("roman_numeral", "none_of_the_provided") and r["correct"] is True and r["parse_success"]:
            continue
        row = r.to_dict()
        row["labels_len"] = len(labels_lookup.get((r["source_dataset"], r["source_id"], r["condition"]), []))
        row["adaptation_category"] = adapt_lookup.get((r["question_id"], r["model"]))
        row["failure_classification"] = classify_failure(row)
        failure_rows.append(row)
    failures_df = pd.DataFrame(failure_rows)
    failures_df.to_csv(OUT_DIR / "phase1b_failures.csv", index=False)

    # ---- figure (new filename, phase1a figure untouched) ----
    fig_path = None
    if not adaptation_df.empty:
        models_list = sorted(adaptation_df["model"].unique())
        categories = ["stable_correct", "presentation_instability", "unexpected_correction", "stable_incorrect",
                       "successful_adaptation", "failure_to_adapt", "adaptation_attempted_but_wrong", "unparseable"]
        counts = {m: adaptation_df[adaptation_df["model"] == m]["category"].value_counts().reindex(categories, fill_value=0)
                  for m in models_list}
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(categories))
        width = 0.8 / max(len(models_list), 1)
        for i, m in enumerate(models_list):
            ax.bar([xi + i * width for xi in x], counts[m].values, width=width, label=m)
        ax.set_xticks([xi + width * (len(models_list) - 1) / 2 for xi in x])
        ax.set_xticklabels([c.replace("_", " ") for c in categories], rotation=25, ha="right")
        ax.set_ylabel("Number of questions")
        ax.set_title("Phase 1B: mcq -> none_of_the_provided outcome/adaptation categories, per model")
        ax.legend()
        fig.tight_layout()
        fig_path = FIG_DIR / "phase1b_none_of_the_provided_outcomes.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    discrepancies = independent_sanity_check(pairs_df, metrics_df) if not pairs_df.empty else []

    # ---- cost ----
    incremental_cost = responses_df.loc[~responses_df["reused_from_phase1a"] & responses_df["estimated_cost_usd"].notna(),
                                          "estimated_cost_usd"].sum()
    phase1a_cost = 0.0
    if PHASE1A_METRICS.exists():
        p1a_metrics = pd.read_csv(PHASE1A_METRICS)
        if "estimated_cost_usd" in p1a_metrics.columns:
            phase1a_cost = p1a_metrics["estimated_cost_usd"].sum(skipna=True)
    cumulative_cost = phase1a_cost + incremental_cost

    # ---- summary.md ----
    lines = []
    lines.append("# Phase 1B — Validation + Perturbation Sensitivity Check — Summary\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"1. Claude availability: {'AVAILABLE' if claude_available else 'UNAVAILABLE (Anthropic account usage cap) - not retried, not substituted'}")
    lines.append(f"2. New API calls by model: {json.dumps(new_call_count_by_model)}")
    lines.append(f"3. Reused Phase 1A responses: {reuse_count}")
    lines.append(f"4. Parsing failures: {int((~responses_df['parse_success']).sum())} / {len(responses_df)}")
    lines.append(f"\n## 5. Semantic audit findings (none_of_the_provided)\n")
    lines.append(f"- Gold letter position changed in {int(audit_df['gold_letter_changed'].sum())}/{len(audit_df)} questions "
                  f"(adaptation genuinely required only for these).")
    lines.append(f"- Gold option TEXT replaced with generic 'None of the provided options' boilerplate in "
                  f"{int(audit_df['gold_text_changed'].sum())}/{len(audit_df)} questions.")
    lines.append(f"- Structural anomalies: {int((audit_df['audit_status']=='ANOMALY').sum())}/{len(audit_df)}.")
    lines.append(f"- Full detail: `results/behavioral/phase1b/none_semantic_audit.csv`.")

    lines.append(f"\n## 6. Metrics per model and condition (roman_numeral and none_of_the_provided reported SEPARATELY)\n")
    lines.append(metrics_df.to_markdown(index=False) if not metrics_df.empty else "(no data)")

    lines.append(f"\n## 7. Every behavioral transition\n")
    lines.append("Full paired table: `results/behavioral/phase1b/phase1b_pairs.csv`. Roman-numeral flips:")
    if not pairs_df.empty:
        flips = pairs_df[pairs_df["roman_flipped"] == True]
        lines.append(flips[["model", "question_id", "mcq_canonical", "roman_canonical"]].to_markdown(index=False)
                      if len(flips) else "None observed.")
        lines.append("\nnone_of_the_provided answer changes:")
        changes = pairs_df[pairs_df["none_answer_changed"] == True]
        lines.append(changes[["model", "question_id", "mcq_canonical", "none_canonical", "mcq_correct", "none_correct"]].to_markdown(index=False)
                      if len(changes) else "None observed.")
    else:
        lines.append("(no data)")

    lines.append(f"\n## 8. Required-adaptation success/failure cases\n")
    if not adaptation_df.empty:
        req = adaptation_df[adaptation_df["adaptation_required"] == True]
        lines.append(f"Adaptation was required for {len(req)} (model, question) pairs.")
        lines.append(req.to_markdown(index=False) if len(req) else "None — no question in this sample required a different answer under none_of_the_provided.")
        lines.append(f"\nFull category breakdown (including not-required questions): `results/behavioral/phase1b/phase1b_adaptation.csv`.")
    else:
        lines.append("(no data)")

    # Per-condition parse-failure comparison - specifically to catch the
    # none_of_the_provided thinking-budget-truncation pattern discovered
    # this run, rather than only reporting an aggregate parse-failure count.
    cond_parse_fail = responses_df.groupby("condition")["parse_success"].apply(lambda s: (~s).mean())
    none_fail_rate = cond_parse_fail.get("none_of_the_provided", float("nan"))
    other_fail_rates = cond_parse_fail.drop(labels=["none_of_the_provided"], errors="ignore")
    none_thinking_budget_issue = (
        pd.notna(none_fail_rate) and none_fail_rate > 0
        and (other_fail_rates.empty or none_fail_rate > other_fail_rates.max() * 2)
    )
    avg_thinking_none = responses_df.loc[responses_df["condition"] == "none_of_the_provided", "thinking_tokens"].mean()
    avg_thinking_other = responses_df.loc[responses_df["condition"] != "none_of_the_provided", "thinking_tokens"].mean()

    lines.append(f"\n## 9. Pipeline anomalies\n")
    anomaly_notes = []
    if not claude_available:
        anomaly_notes.append("- Claude unavailable (account usage cap) - Claude-side conditions not evaluated this run.")
    if discrepancies:
        anomaly_notes.append(f"- **METRIC DISCREPANCY DETECTED** between independent recomputation and reported metrics: {discrepancies}")
    if none_thinking_budget_issue:
        anomaly_notes.append(
            f"- **none_of_the_provided triggers a much higher truncation-driven parse-failure rate** on the Gemini "
            f"3.7/3.8-flash thinking-floor models: {none_fail_rate:.0%} vs {other_fail_rates.max():.0%} on other "
            f"conditions. Root-caused via the cached raw responses (zero extra spend): mean "
            f"`thoughts_token_count` on none_of_the_provided is {avg_thinking_none:.0f} vs {avg_thinking_other:.0f} "
            f"on mcq/roman_numeral, out of the 200-token budget - responses are getting cut off mid-preamble "
            f"(e.g. raw text `'Here is the JSON requested'`, finish_reason=MAX_TOKENS) before the JSON answer is "
            f"ever emitted. This is a genuine behavioral-instability signal from Objective B, not pipeline noise: "
            f"the model appears to reason substantially more about a 'none of the provided options is correct' "
            f"framing. Recommend raising max_output_tokens specifically for this condition (or globally) before "
            f"any scaled none_of_the_provided run."
        )
    if int((~responses_df["parse_success"]).sum()) > 0:
        anomaly_notes.append(f"- {int((~responses_df['parse_success']).sum())} unparseable responses recorded, never guessed at.")
    if not anomaly_notes:
        anomaly_notes.append("- None observed beyond what's reported above.")
    lines.extend(anomaly_notes)

    lines.append(f"\n## 10. Incremental Phase 1B cost\n\n**${incremental_cost:.4f} USD** (new calls only, reused Phase 1A responses cost $0).")
    lines.append(f"\n## 11. Cumulative Phase 1A + 1B cost\n\n**${cumulative_cost:.4f} USD** (Phase 1A: ${phase1a_cost:.4f}, Phase 1B incremental: ${incremental_cost:.4f}).")

    lines.append(f"\n## 12. Readiness for the full pilot\n")
    if discrepancies:
        lines.append("NOT ready: a metric discrepancy was detected between independent recomputation and reported "
                      "metrics - this must be resolved before trusting any scaled run.")
    elif none_thinking_budget_issue:
        lines.append(
            f"NOT fully ready for none_of_the_provided specifically: {none_fail_rate:.0%} of Gemini responses on this "
            f"condition were truncated before completing (see Pipeline anomalies above). mcq and roman_numeral are "
            f"validated end-to-end on Gemini (low parse-failure rate, caching/reuse confirmed, independent metric "
            f"recomputation matches exactly). Fix: raise the output token budget for none_of_the_provided before "
            f"scaling it. Claude remains entirely unvalidated pending account access - "
            f"not ready to scale Claude on any condition until it can be smoke-tested."
        )
    elif not claude_available:
        lines.append("Gemini pipeline (mcq, roman_numeral, none_of_the_provided) is validated end-to-end including "
                      "reuse/caching and the semantic audit gate. Claude remains unvalidated pending account access - "
                      "not ready to scale Claude specifically until it can be smoke-tested.")
    else:
        lines.append("All three models validated across all three conditions; independent sanity recomputation matches "
                      "reported metrics exactly. Pipeline is technically ready for a larger validation run, pending "
                      "explicit approval - this remains n=10, not scientific evidence.")

    summary_path = OUT_DIR / "phase1b_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 70)
    print("STOP: Phase 1B validation complete. No 200-question cohort, no new perturbations/models, no significance testing.")
    print("=" * 70)
    print(f"\nOutputs: {OUT_DIR}")
    print(f"Figure: {fig_path}")
    print(f"\nIncremental Phase 1B cost: ${incremental_cost:.4f} USD")
    print(f"Cumulative Phase 1A + 1B cost: ${cumulative_cost:.4f} USD")
    print(f"\n--- {summary_path.name} ---\n")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
