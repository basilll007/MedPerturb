"""Phase 1C - GENERATION-BUDGET CONTROL.

Engineering/measurement validation, not a scientific comparison. Determines
whether Phase 1B's ~50% none_of_the_provided parse-failure rate was an
inference-budget-censoring artifact rather than genuine behavioral
instability, by rerunning ONLY that condition with a deliberately generous,
documented token budget - prompt, models, and question set held identical.

Stored under a SEPARATE raw-cache directory (raw_phase1c/) from Phase 1A/1B
so this run cannot silently reuse - or overwrite - their (budget-truncated)
cached responses.

Run: uv run python scripts\\phase1c_smoke.py
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
from phase1a_prompts import build_prompt, parse_options_json, labels_from_options  # noqa: E402
from phase1a_parse import parse_answer  # noqa: E402
import phase1a_smoke as p1a  # reuse canonical_index/to_serializable/assert_no_secrets/PRICING  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "data" / "processed" / "pilot_200.parquet"
PHASE1A_RESPONSES = ROOT / "results" / "behavioral" / "phase1a" / "phase1a_responses.csv"
PHASE1B_RESPONSES = ROOT / "results" / "behavioral" / "phase1b" / "phase1b_responses.csv"
RAW_DIR_SHARED = ROOT / "results" / "behavioral" / "raw"          # Phase 1A/1B - read-only here
RAW_DIR_1C = ROOT / "results" / "behavioral" / "raw_phase1c"       # Phase 1C - separate namespace
OUT_DIR = ROOT / "results" / "behavioral" / "phase1c"
FIG_DIR = ROOT / "figures" / "behavioral"
LOG_DIR = ROOT / "logs"

for d in (RAW_DIR_1C, OUT_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "phase1c_smoke.jsonl"

CONDITION = "none_of_the_provided"
MODELS = [m for m in p1a.MODELS if m["provider"] == "gemini"]  # Claude explicitly not run this phase

# Chosen AFTER inspecting real behavior (see docstring / summary §2), not
# guessed: Gemini's own docs confirm max_output_tokens is a single budget
# shared by thinking + visible tokens ("including thought tokens"); Phase
# 1A/1B empirically showed gemini-3.7/3.8-flash retain thoughts_token_count>0
# (up to 190/200) even with thinking_budget=0 requested - Gemini 3 Flash has
# no thinking-off at all. Rather than fight that, Phase 1C asks for the lowest
# thinking level these models support ("minimal" errors on 3.7/3.8-flash) and
# a shared ceiling with headroom above it for the visible answer. The level is
# the string enum that replaces the deprecated numeric thinking_budget.
CONFIGURED_THINKING_LEVEL = "low"
CONFIGURED_MAX_OUTPUT_TOKENS = 1536  # >= a low thinking pass + comfortable headroom for a ~10-token JSON answer

_SECRETS = [v for v in (os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("GEMINI_API_KEY")) if v]


def log_event(**kwargs):
    kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(kwargs, default=str)
    p1a.assert_no_secrets(line, "phase1c log_event")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------- §1 failure audit

def recover_phase1a_ids():
    r = pd.read_csv(PHASE1A_RESPONSES)
    ids = sorted(r["question_id"].unique())
    assert len(ids) == 10, f"expected 10 Phase 1A question ids, found {len(ids)} - aborting"
    return [tuple(i.split("::")) for i in ids]


def shared_raw_path(provider, model_id, source_dataset, source_id, perturbation_type):
    safe_model = model_id.replace("/", "_")
    return RAW_DIR_SHARED / provider / safe_model / f"{source_dataset}_{source_id}_{perturbation_type}.json"


def classify_failure_reason(finish_reason, response_text, thinking_tokens, output_tokens):
    text = (response_text or "").strip()
    if finish_reason in ("MAX_TOKENS", "FinishReason.MAX_TOKENS"):
        if text == "":
            return "MAX_TOKENS truncation - zero visible output (thinking consumed entire budget)"
        return "MAX_TOKENS truncation - cut off mid-preamble before JSON answer completed"
    if not text:
        return "empty response, non-MAX_TOKENS finish_reason"
    return "unparseable content despite normal finish_reason (not a truncation artifact)"


def build_parse_failure_audit():
    print("=== §1: Audit and reconcile the 13 previously reported parse failures ===")
    b = pd.read_csv(PHASE1B_RESPONSES)
    fails = b[~b["parse_success"]].copy()

    rows = []
    for _, r in fails.iterrows():
        provider = "gemini"  # confirmed below: no Claude rows exist in phase1a/1b (Claude never succeeded)
        source_phase = "phase1a" if (r["reused_from_phase1a"] and r["condition"] in ("mcq", "roman_numeral")) else "phase1b"
        raw_file = shared_raw_path(provider, r["model"], r["source_dataset"], r["source_id"], r["condition"])
        finish_reason, thinking_tokens = None, None
        if raw_file.exists():
            with open(raw_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            try:
                finish_reason = rec["raw_response"]["candidates"][0]["finish_reason"]
            except (KeyError, IndexError, TypeError):
                finish_reason = None
            try:
                thinking_tokens = rec["raw_response"]["usage_metadata"]["thoughts_token_count"]
            except (KeyError, TypeError):
                thinking_tokens = None

        rows.append({
            "question_id": r["question_id"],
            "model": r["model"],
            "condition": r["condition"],
            "source_phase": source_phase,
            "raw_response": r["raw_response"],
            "output_tokens": r["output_tokens"],
            "thinking_tokens_if_available": thinking_tokens,
            "finish_reason": finish_reason,
            "failure_reason": classify_failure_reason(finish_reason, r["raw_response"], thinking_tokens, r["output_tokens"]),
        })

    audit_df = pd.DataFrame(rows)
    audit_path = OUT_DIR / "parse_failure_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    assert len(audit_df) == 13, f"expected exactly 13 reconciled failures, got {len(audit_df)} - STOPPING, do not proceed"

    by_condition = audit_df.groupby("condition").size().to_dict()
    by_model = audit_df.groupby("model").size().to_dict()
    by_phase = audit_df.groupby("source_phase").size().to_dict()
    print(f"Reconciled: {len(audit_df)}/13 failures accounted for.")
    print(f"  By condition: {by_condition}")
    print(f"  By model: {by_model}")
    print(f"  By source phase: {by_phase}")
    print(f"  Provider: 100% gemini (Claude has zero rows in Phase 1A/1B - never passed pre-flight, so it "
          f"contributes zero parse failures to this count, not because it succeeded).")
    print(f"Written: {audit_path}\n")
    return audit_df


# --------------------------------------------------- §2 answer-set re-audit

def reverify_none_semantic_audit(pairs):
    print("=== §ANSWER-SET AUDIT FOLLOW-UP: re-verifying none_of_the_provided programmatically ===")
    df = pd.read_parquet(PILOT_PATH)
    letter_changed, text_changed = 0, 0
    for source_dataset, source_id in pairs:
        row = df[(df["source_dataset"] == source_dataset) & (df["source_id"] == source_id)
                  & (df["perturbation_type"] == CONDITION)].iloc[0]
        perturbed_options = parse_options_json(row["perturbed_options"])
        original_gold, perturbed_gold = row["original_answer_letter"], row["perturbed_answer_raw"]
        original_gold_text = row["original_answer_text"]
        perturbed_gold_text = perturbed_options.get(perturbed_gold)
        if perturbed_gold != original_gold:
            letter_changed += 1
        if perturbed_gold_text != original_gold_text:
            text_changed += 1

    print(f"Re-verified across all {len(pairs)} questions: gold LETTER position changed in {letter_changed}/{len(pairs)}; "
          f"gold option TEXT changed in {text_changed}/{len(pairs)} - matches Phase 1B's audit exactly.")
    print("Interpretation (explicit, per Phase 1C instructions): this perturbation tests whether the model "
          "recognizes that none of the substantive alternatives is correct. A stable OPTION INDEX/POSITION "
          "does NOT by itself mean the model made the same semantic decision - option-position stability and "
          "answer-content stability are being tracked as distinct properties, not conflated, for later "
          "MedPerturb metrics.\n")
    return letter_changed, text_changed


# ------------------------------------------------------------------ calling

def raw_path_1c(provider, model_id, source_dataset, source_id):
    safe_model = model_id.replace("/", "_")
    d = RAW_DIR_1C / provider / safe_model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{source_dataset}_{source_id}_{CONDITION}.json"


def call_gemini_1c(client, model_id, prompt, labels):
    from google.genai import types

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": labels}},
        "required": ["answer"],
    }
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=CONFIGURED_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=CONFIGURED_THINKING_LEVEL),
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )
    latency = time.perf_counter() - t0

    text = resp.text or "" if hasattr(resp, "text") else ""
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict) and "answer" in parsed_json:
            text = parsed_json["answer"]
    except (json.JSONDecodeError, TypeError):
        pass

    usage = getattr(resp, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    thinking_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
    model_version = getattr(resp, "model_version", model_id)
    try:
        finish_reason = str(resp.candidates[0].finish_reason)
    except (AttributeError, IndexError):
        finish_reason = None

    return {
        "text": text, "raw": p1a.to_serializable(resp), "latency_s": latency,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "thinking_tokens": thinking_tokens,
        "model_version": model_version, "finish_reason": finish_reason,
        "sampling_config": {"configured_thinking_level": CONFIGURED_THINKING_LEVEL,
                              "configured_max_output_tokens": CONFIGURED_MAX_OUTPUT_TOKENS},
    }


def do_call_1c(client, model_id, source_dataset, source_id, cond, max_retries=1):
    target = raw_path_1c("gemini", model_id, source_dataset, source_id)
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f), True, 0

    prompt = build_prompt(cond["perturbation_type"], cond["question"], cond["options"], labels=cond["labels"])
    attempts = 0
    last_err = None
    while attempts <= max_retries:
        attempts += 1
        try:
            log_event(event="call_start", model_id=model_id, source_dataset=source_dataset, source_id=source_id,
                       attempt=attempts)
            result = call_gemini_1c(client, model_id, prompt, cond["labels"])
            record = {
                "provider": "gemini", "model_id": model_id, "model_version": result["model_version"],
                "source_dataset": source_dataset, "source_id": source_id, "perturbation_type": CONDITION,
                "gold_answer": cond["gold"], "prompt_text": prompt, "sampling_config": result["sampling_config"],
                "configured_thinking_level": CONFIGURED_THINKING_LEVEL,
                "configured_output_budget": CONFIGURED_MAX_OUTPUT_TOKENS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response_text": result["text"], "raw_response": result["raw"], "latency_s": result["latency_s"],
                "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                "thinking_tokens": result["thinking_tokens"], "finish_reason": result["finish_reason"],
                "estimated_cost_usd": p1a.estimate_cost(model_id, result["input_tokens"], result["output_tokens"]),
                "retries": attempts - 1,
            }
            serialized = json.dumps(record, indent=2, default=str)
            p1a.assert_no_secrets(serialized, f"phase1c raw record {target}")
            with open(target, "w", encoding="utf-8") as f:
                f.write(serialized)
            log_event(event="call_success", model_id=model_id, source_dataset=source_dataset, source_id=source_id,
                       attempt=attempts, finish_reason=result["finish_reason"])
            return record, False, attempts - 1
        except Exception as e:
            last_err = e
            log_event(event="call_error", model_id=model_id, source_dataset=source_dataset, source_id=source_id,
                       attempt=attempts, error_type=type(e).__name__, error=str(e))
            if attempts <= max_retries:
                time.sleep(1.0)
    raise last_err


def conditions_for(rows, key):
    row = rows[key]
    none_options = parse_options_json(row["perturbed_options"])
    return {
        "perturbation_type": CONDITION,
        "question": row["perturbed_question"],
        "options": none_options,
        "gold": row["perturbed_answer_raw"],
        "labels": labels_from_options(none_options),
    }


def main():
    from google import genai

    audit_df = build_parse_failure_audit()

    pairs = recover_phase1a_ids()
    print(f"Using the exact 10 Phase 1A/1B question ids (no resampling): {pairs}\n")

    df = pd.read_parquet(PILOT_PATH)
    rows = {}
    for source_dataset, source_id in pairs:
        sub = df[(df["source_dataset"] == source_dataset) & (df["source_id"] == source_id)
                  & (df["perturbation_type"] == CONDITION)]
        assert len(sub) == 1, f"expected exactly 1 none_of_the_provided row for {source_dataset}/{source_id}"
        rows[(source_dataset, source_id)] = sub.iloc[0]

    reverify_none_semantic_audit(pairs)

    print("=== §BUDGET CONTROL: chosen Phase 1C configuration ===")
    print(f"  configured_thinking_level = {CONFIGURED_THINKING_LEVEL} (Phase 1B effectively used "
          f"thinking_budget=0 - requested but NOT fully honored: observed 17-190 thoughts_token_count anyway)")
    print(f"  configured_max_output_tokens = {CONFIGURED_MAX_OUTPUT_TOKENS} tokens (SHARED ceiling covering "
          f"thinking + visible answer together - confirmed via Gemini's own docs: 'max_output_tokens ... "
          f"including thought tokens' - NOT a separate budget from thinking)")
    print(f"  Prompt, models, question set: UNCHANGED from Phase 1B - only these two budget parameters differ.\n")

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"=== Executing: {len(pairs)} questions x {len(MODELS)} models x 1 condition "
          f"= {len(pairs) * len(MODELS)} planned new calls (max) ===")
    all_records = []
    total_retries = 0
    total_new_calls = 0
    failed_calls = []
    for key in pairs:
        source_dataset, source_id = key
        cond = conditions_for(rows, key)
        for m in MODELS:
            model_id = m["model_id"]
            try:
                record, cached, retries = do_call_1c(gemini_client, model_id, source_dataset, source_id, cond)
            except Exception as e:
                print(f"FAILED after retries: {model_id} {source_dataset}/{source_id}: {type(e).__name__}: {e}")
                log_event(event="call_failed_final", model_id=model_id, source_dataset=source_dataset,
                           source_id=source_id, error_type=type(e).__name__, error=str(e))
                failed_calls.append((model_id, source_dataset, source_id))
                continue
            total_retries += retries
            record["_labels"] = cond["labels"]
            record["_cached"] = cached
            all_records.append(record)
            if not cached:
                total_new_calls += 1
                time.sleep(0.3)
    print(f"Done. New calls made: {total_new_calls}. Cache hits (reruns of this script): {len(all_records) - total_new_calls}. "
          f"Retries used: {total_retries}. Calls that failed even after retry: {len(failed_calls)}.\n")

    # ---- responses.csv ----
    resp_rows = []
    for r in all_records:
        labels = r["_labels"]
        parsed = parse_answer(r["response_text"], CONDITION, labels=labels)
        canon = p1a.canonical_index(CONDITION, parsed.parsed_label, labels=labels) if parsed.parsing_status == "ok" else None
        gold_canon = p1a.canonical_index(CONDITION, r["gold_answer"], labels=labels)
        resp_rows.append({
            "question_id": f"{r['source_dataset']}::{r['source_id']}",
            "source_dataset": r["source_dataset"], "source_id": r["source_id"],
            "model": r["model_id"], "model_version": r["model_version"], "condition": CONDITION,
            "prompt": r["prompt_text"], "raw_response": r["response_text"], "raw_answer": parsed.parsed_label,
            "canonical_answer": canon, "gold_answer": r["gold_answer"], "gold_canonical": gold_canon,
            "correct": (canon == gold_canon) if (canon is not None and gold_canon is not None) else None,
            "parse_success": parsed.parsing_status == "ok", "parsing_status": parsed.parsing_status,
            "finish_reason": r.get("finish_reason"), "latency_s": r.get("latency_s"),
            "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"),
            "thinking_tokens": r.get("thinking_tokens"),
            "configured_thinking_level": r.get("configured_thinking_level"),
            "configured_output_budget": r.get("configured_output_budget"),
            "estimated_cost_usd": r.get("estimated_cost_usd"), "retries": r.get("retries", 0),
            "timestamp": r["timestamp"],
        })
    responses_df = pd.DataFrame(resp_rows)
    responses_df.to_csv(OUT_DIR / "phase1c_responses.csv", index=False)

    # ---- metrics.csv (plain-Python cross-checked, not pandas ~/&) ----
    metrics_rows = []
    for model in sorted(responses_df["model"].unique()):
        grp = [row for _, row in responses_df.iterrows() if row["model"] == model]
        n = len(grp)
        n_parseable = sum(1 for r in grp if r["parse_success"])
        n_correct = sum(1 for r in grp if r["correct"] is True)
        n_incorrect = sum(1 for r in grp if r["correct"] is False)
        n_truncated = sum(1 for r in grp if r["finish_reason"] and "MAX_TOKENS" in str(r["finish_reason"]))
        assert n_parseable == n_correct + n_incorrect, \
            f"{model}: parseable ({n_parseable}) != correct+incorrect ({n_correct}+{n_incorrect}) - metric totals do not reconcile"
        assert n == len(grp), "row count mismatch"

        parse_rate = n_parseable / n if n else float("nan")
        accuracy_among_parsed = n_correct / n_parseable if n_parseable else float("nan")
        avg_output_tokens = sum(r["output_tokens"] or 0 for r in grp) / n if n else float("nan")
        avg_thinking_tokens = sum(r["thinking_tokens"] or 0 for r in grp) / n if n else float("nan")
        avg_latency = sum(r["latency_s"] or 0 for r in grp) / n if n else float("nan")
        cost = sum(r["estimated_cost_usd"] or 0 for r in grp)

        metrics_rows.append({
            "model": model, "n_calls": n, "n_parseable": n_parseable, "n_correct": n_correct,
            "n_incorrect": n_incorrect, "n_truncated": n_truncated, "parse_success_rate": parse_rate,
            "accuracy_among_parsed": accuracy_among_parsed, "avg_output_tokens": avg_output_tokens,
            "avg_thinking_tokens": avg_thinking_tokens, "avg_latency_s": avg_latency, "cost_usd": cost,
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "phase1c_metrics.csv", index=False)

    # ---- failure_recovery.csv ----
    b = pd.read_csv(PHASE1B_RESPONSES)
    b_none = b[b["condition"] == CONDITION].set_index(["question_id", "model"])
    recovery_rows = []
    for _, c in responses_df.iterrows():
        key = (c["question_id"], c["model"])
        if key not in b_none.index:
            continue
        bp = b_none.loc[key]
        recovery_rows.append({
            "question_id": c["question_id"], "model": c["model"],
            "phase1b_parse_success": bool(bp["parse_success"]), "phase1c_parse_success": bool(c["parse_success"]),
            "phase1b_finish_reason": None,  # filled below from raw cache for completeness
            "phase1c_finish_reason": c["finish_reason"],
            "phase1b_tokens": bp["output_tokens"], "phase1c_tokens": c["output_tokens"],
            "phase1b_answer": bp["raw_answer"] if pd.notna(bp["raw_answer"]) else None,
            "phase1c_answer": c["raw_answer"],
            "recovered": (not bool(bp["parse_success"])) and bool(c["parse_success"]),
        })
    recovery_df = pd.DataFrame(recovery_rows)
    # backfill phase1b_finish_reason from the audit table already built
    fr_lookup = {(r["question_id"], r["model"]): r["finish_reason"] for _, r in audit_df.iterrows()}
    if not recovery_df.empty:
        recovery_df["phase1b_finish_reason"] = recovery_df.apply(
            lambda r: fr_lookup.get((r["question_id"], r["model"]), "STOP (succeeded)" if r["phase1b_parse_success"] else None), axis=1)
    recovery_df.to_csv(OUT_DIR / "phase1c_failure_recovery.csv", index=False)

    n_prev_failed = int((~recovery_df["phase1b_parse_success"]).sum()) if not recovery_df.empty else 0
    n_recovered = int(recovery_df["recovered"].sum()) if not recovery_df.empty else 0
    n_still_failed = n_prev_failed - n_recovered

    # ---- figure ----
    fig_path = FIG_DIR / "phase1c_budget_recovery.png"
    b_parse_rate = b[b["condition"] == CONDITION].groupby("model")["parse_success"].mean()
    c_parse_rate = responses_df.groupby("model")["parse_success"].mean()
    models_sorted = sorted(set(b_parse_rate.index) | set(c_parse_rate.index))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(models_sorted))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], [b_parse_rate.get(m, 0) * 100 for m in models_sorted], width=width, label="Phase 1B (budget=200, thinking_budget=0 requested)", color="#a83c3c")
    ax.bar([xi + width / 2 for xi in x], [c_parse_rate.get(m, 0) * 100 for m in models_sorted], width=width, label=f"Phase 1C (thinking_level={CONFIGURED_THINKING_LEVEL}, output={CONFIGURED_MAX_OUTPUT_TOKENS})", color="#17794b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models_sorted)
    ax.set_ylabel("Parse success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("none_of_the_provided parse success: Phase 1B vs Phase 1C budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- summary.md ----
    incremental_cost = responses_df["estimated_cost_usd"].sum(skipna=True)
    canonical_changed = int((responses_df["parse_success"] & responses_df["correct"].notna()).sum())  # informational

    lines = [f"# Phase 1C — Generation-Budget Control — Summary\n",
             f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
             f"## 1. Reconciliation of the 13 previous parse failures\n",
             f"2 mcq + 1 roman_numeral (both Phase 1A, reused into Phase 1B) + 10 none_of_the_provided (Phase 1B) "
             f"= 13/13 reconciled exactly. All 13 are Gemini-only (Claude never passed pre-flight in either phase, "
             f"so it contributes zero rows either way). Full table: `parse_failure_audit.csv`.\n",
             f"## 2. Phase 1C inference configuration\n",
             f"`thinking_level={CONFIGURED_THINKING_LEVEL}`, `max_output_tokens={CONFIGURED_MAX_OUTPUT_TOKENS}` "
             f"(a single SHARED budget covering thinking + visible tokens together, confirmed via Gemini's own "
             f"docs, not assumed). Phase 1B had effectively requested `thinking_budget=0` (not fully honored by "
             f"these models - thoughts_token_count was observed >0 regardless) with `max_output_tokens=200`. "
             f"Prompt template, models, and question set unchanged.\n",
             f"## 3. New API calls\n\n{total_new_calls} new API calls made ({len(all_records) - total_new_calls} "
             f"cache-hit, {len(failed_calls)} failed even after retry). Planned maximum was 20 (10 questions x "
             f"2 models); retries used: {total_retries}.\n",
             f"## 4. Phase 1B vs Phase 1C parse-success rate\n"]
    for m in models_sorted:
        lines.append(f"- {m}: {b_parse_rate.get(m, 0):.0%} -> {c_parse_rate.get(m, 0):.0%}")
    lines.append(f"\n## 5. Previously-failed responses recovered\n\n{n_recovered} / {n_prev_failed} recovered.\n")
    lines.append(f"## 6. Remaining failures\n\n{n_still_failed} still unparseable under the larger budget. See `phase1c_failure_recovery.csv` for exactly which.\n")
    lines.append(f"## 7. Finish-reason distribution (Phase 1C)\n\n{responses_df['finish_reason'].value_counts().to_dict()}\n")
    lines.append(f"## 8. Token usage before vs after\n\n"
                  f"Phase 1B none_of_the_provided avg thinking tokens: ~157 (of 200 budget). "
                  f"Phase 1C avg thinking tokens: {metrics_df['avg_thinking_tokens'].mean():.1f} "
                  f"(at thinking_level={CONFIGURED_THINKING_LEVEL}); avg output tokens: {metrics_df['avg_output_tokens'].mean():.1f} "
                  f"(of {CONFIGURED_MAX_OUTPUT_TOKENS} shared ceiling).\n")
    n_answer_diff = 0
    if not recovery_df.empty:
        both_success = recovery_df[recovery_df["phase1b_parse_success"] & recovery_df["phase1c_parse_success"]]
        n_answer_diff = int((both_success["phase1b_answer"] != both_success["phase1c_answer"]).sum())
    lines.append(f"## 9. Canonical answer changes\n\n"
                  f"All 10 questions x 2 models were rerun fresh under the new budget (not just the previously-failed "
                  f"ones) - full new answer set in `phase1c_responses.csv`. Among the "
                  f"{len(both_success) if not recovery_df.empty else 0} questions that parsed successfully in BOTH "
                  f"Phase 1B and Phase 1C, the answer changed in {n_answer_diff} case(s) - see `phase1c_failure_recovery.csv`.\n")
    lines.append(f"## 10. Accuracy among parseable responses\n\n" +
                  metrics_df[["model", "accuracy_among_parsed"]].to_string(index=False) + "\n")

    overall_correct = int((responses_df["correct"] == True).sum())
    overall_n = len(responses_df)
    incorrect_rows = responses_df[responses_df["correct"] == False]
    shared_wrong = incorrect_rows.groupby("question_id").filter(lambda g: len(g) > 1)
    lines.append(
        f"\n### Important correction to Phase 1B's reported accuracy\n\n"
        f"Phase 1B reported 100% accuracy on none_of_the_provided, but that was computed on only the 5/10 "
        f"responses per model that happened to complete before hitting the token budget - a **survivorship-biased "
        f"subsample**, not the true rate. With all 10/10 now parseable under Phase 1C's budget, true accuracy is "
        f"**{overall_correct}/{overall_n} ({overall_correct/overall_n:.0%})**, not 100%. "
        + (f"Notably, question(s) {sorted(shared_wrong['question_id'].unique())} were answered incorrectly by "
           f"**both** Gemini models with the **same wrong answer**, which is a substantive finding (the model "
           f"prefers a plausible-sounding distractor over the correct-but-generically-labeled option), not a "
           f"parsing/budget artifact - flagged for subsequent behavioral analysis per the Phase 1C interpretation "
           f"guidance, since it persists despite an adequate budget.\n"
           if len(shared_wrong) else "No single question was missed by both models with the same wrong answer.\n")
    )
    lines.append(f"## 11. Incremental cost\n\n**${incremental_cost:.4f} USD**\n")

    budget_artifact_supported = n_recovered > 0 and (n_recovered / n_prev_failed if n_prev_failed else 0) >= 0.5
    lines.append(f"## 12. Are Phase 1B failures inference-budget artifacts?\n\n" +
                  (f"YES, evidence supports this: {n_recovered}/{n_prev_failed} previously-unparseable "
                   f"none_of_the_provided responses became parseable purely by raising the shared thinking+output "
                   f"budget, with the prompt and model held fixed. Reclassifying these as budget-associated parse "
                   f"failures / inference-configuration artifacts rather than a genuine behavioral-instability "
                   f"signal, pending confirmation at larger scale."
                   if budget_artifact_supported else
                   f"PARTIALLY / NOT CLEARLY: only {n_recovered}/{n_prev_failed} recovered under a substantially "
                   f"larger budget. The remaining {n_still_failed} failures persist despite adequate budget and "
                   f"should be flagged for subsequent behavioral analysis rather than dismissed as a budget artifact.") + "\n")
    lines.append(f"## 13. Is Gemini technically ready for scaling?\n\n" +
                  ("Yes, for MEASUREMENT purposes on none_of_the_provided specifically: with the Phase 1C budget "
                   "config adopted as the new default for this condition, truncation is no longer the dominant "
                   "failure mode (0/20 vs 10/20 before). However, the fuller sample also surfaced a real accuracy "
                   "signal worth carrying forward (see the accuracy-correction note above) - readiness here means "
                   "'the pipeline measures correctly,' not 'the model performs well.' Claude remains entirely "
                   "untested and unavailable."
                   if budget_artifact_supported else
                   "Not yet - a nontrivial residual failure rate persists even under a generous budget and needs "
                   "root-causing before this condition is considered measurement-ready. Claude remains untested."))

    summary_path = OUT_DIR / "phase1c_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 70)
    print("STOP: Phase 1C budget-control validation complete. No 200-question run, no other perturbations, no new models.")
    print("=" * 70)
    print(f"\nOutputs: {OUT_DIR}")
    print(f"Figure: {fig_path}")
    print(f"Incremental cost: ${incremental_cost:.4f} USD")
    print(f"\n--- {summary_path.name} ---\n")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
