"""Phase 1A - CONTROLLED BEHAVIORAL PILOT - SMOKE TEST (v2).

10 fixed pilot-cohort questions x 2 conditions (mcq, roman_numeral) x 3 models
= 60 planned API calls maximum. Canonical answer-space scoring (both label
systems mapped to option index 0-3, never string-compared directly).
Pre-flight: one sample call per model verified before the batch runs.

Run: uv run python scripts\\phase1a_smoke.py
"""

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1a_prompts import PROMPT_VERSION, build_prompt, parse_options_json, get_labels  # noqa: E402
from phase1a_parse import parse_answer  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "data" / "processed" / "pilot_200.parquet"
RAW_DIR = ROOT / "results" / "behavioral" / "raw"
OUT_DIR = ROOT / "results" / "behavioral" / "phase1a"
FIG_DIR = ROOT / "figures" / "behavioral"
LOG_DIR = ROOT / "logs"

SMOKE_N = 10
# 20 was too small: gemini-3.7/3.8-flash have a mandatory minimum internal
# "thinking" budget (observed thoughts_token_count 17-52 even with
# thinking_budget=0 requested) that must be paid for before any visible
# answer token is produced. 200 covers that with headroom; still negligible
# cost (~150 output tokens max at $3.75-5/MTok is a fraction of a cent/call).
MAX_TOKENS = 200

MODELS = [
    {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
    {"provider": "gemini", "model_id": "gemini-3.8-flash"},
    {"provider": "gemini", "model_id": "gemini-3.7-flash"},
]

# Sourced 2026-09-29 via WebFetch of claude.com/pricing and
# ai.google.dev/gemini-api/docs/pricing - not guessed. USD per 1M tokens.
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "source": "claude.com/pricing (2026-09-29)"},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "source": "ai.google.dev/gemini-api/docs/pricing (2026-09-29, promo through 2026-12-31)"},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75, "source": "ai.google.dev/gemini-api/docs/pricing (2026-09-29, promo through 2026-12-31)"},
}

for d in (RAW_DIR, OUT_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "phase1a_smoke_v2.jsonl"

_SECRETS = [v for v in (os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("GEMINI_API_KEY")) if v]


def assert_no_secrets(text: str, context: str):
    for s in _SECRETS:
        if s and s in text:
            raise RuntimeError(f"SECRET LEAK DETECTED in {context} - aborting before write")


def log_event(**kwargs):
    kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(kwargs, default=str)
    assert_no_secrets(line, "log_event")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def to_serializable(obj):
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            try:
                return obj.model_dump()
            except Exception:
                pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def canonical_index(perturbation_type: str, label: str, labels: list[str] | None = None) -> int | None:
    """`labels` overrides the static lookup - required for perturbations
    (e.g. none_of_the_provided) whose valid label set must come from the
    actual example, not an assumed fixed set."""
    if labels is None:
        labels = get_labels(perturbation_type)
    label = (label or "").strip().upper()
    if label not in labels:
        return None
    return labels.index(label)


def raw_path(provider, model_id, source_dataset, source_id, perturbation_type):
    safe_model = model_id.replace("/", "_")
    d = RAW_DIR / provider / safe_model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{source_dataset}_{source_id}_{perturbation_type}.json"


def call_anthropic(client, model_id, prompt, labels):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": labels}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": schema},
        },
    )
    latency = time.perf_counter() - t0

    text = ""
    if resp.content and len(resp.content) > 0:
        text = getattr(resp.content[0], "text", "") or ""
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict) and "answer" in parsed_json:
            text = parsed_json["answer"]
    except (json.JSONDecodeError, TypeError):
        pass

    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    model_version = getattr(resp, "model", model_id)

    return {
        "text": text,
        "raw": to_serializable(resp),
        "latency_s": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_version": model_version,
        "sampling_config": {"temperature": "unsupported_by_api", "effort": "low"},
    }


def call_gemini(client, model_id, prompt, labels):
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
            temperature=0,
            max_output_tokens=MAX_TOKENS,
            # Some Gemini models (e.g. gemini-3.7-flash) enable hidden
            # reasoning by default, which both burns the token budget before
            # any visible answer (observed: finish_reason=MAX_TOKENS, empty
            # text, thoughts_token_count>0) and violates the no-CoT
            # requirement. Disabling explicitly rather than just raising
            # max_output_tokens and hoping.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
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
    model_version = getattr(resp, "model_version", model_id)

    return {
        "text": text,
        "raw": to_serializable(resp),
        "latency_s": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_version": model_version,
        "sampling_config": {"temperature": 0},
    }


def estimate_cost(model_id, input_tokens, output_tokens):
    if model_id not in PRICING or input_tokens is None or output_tokens is None:
        return None
    p = PRICING[model_id]
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


def call_model(clients, provider, model_id, prompt, perturbation_type, labels=None):
    if labels is None:
        labels = get_labels(perturbation_type)
    if provider == "anthropic":
        return call_anthropic(clients["anthropic"], model_id, prompt, labels)
    return call_gemini(clients["gemini"], model_id, prompt, labels)


def do_call_and_cache(clients, provider, model_id, source_dataset, source_id, cond):
    target = raw_path(provider, model_id, source_dataset, source_id, cond["perturbation_type"])
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f), True

    # cond["labels"] (optional) lets callers - e.g. Phase 1B's
    # none_of_the_provided - supply a label set derived from the actual
    # example instead of relying on the static mcq/roman_numeral lookup.
    labels = cond.get("labels")
    prompt = build_prompt(cond["perturbation_type"], cond["question"], cond["options"], labels=labels)
    log_event(event="call_start", provider=provider, model_id=model_id, source_dataset=source_dataset,
               source_id=source_id, perturbation_type=cond["perturbation_type"])
    result = call_model(clients, provider, model_id, prompt, cond["perturbation_type"], labels=labels)

    record = {
        "provider": provider,
        "model_id": model_id,
        "model_version": result["model_version"],
        "source_dataset": source_dataset,
        "source_id": source_id,
        "perturbation_type": cond["perturbation_type"],
        "gold_answer": cond["gold"],
        "prompt_version": PROMPT_VERSION,
        "prompt_text": prompt,
        "sampling_config": result["sampling_config"],
        "max_tokens": MAX_TOKENS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response_text": result["text"],
        "raw_response": result["raw"],
        "latency_s": result["latency_s"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "estimated_cost_usd": estimate_cost(model_id, result["input_tokens"], result["output_tokens"]),
    }
    serialized = json.dumps(record, indent=2, default=str)
    assert_no_secrets(serialized, f"raw record {target}")
    with open(target, "w", encoding="utf-8") as f:
        f.write(serialized)

    log_event(event="call_success", provider=provider, model_id=model_id, source_dataset=source_dataset,
               source_id=source_id, perturbation_type=cond["perturbation_type"], response_text=result["text"],
               latency_s=result["latency_s"])
    return record, False


def build_conditions(row):
    return [
        {
            "perturbation_type": "mcq",
            "question": row["original_question"],
            "options": parse_options_json(row["original_options"]),
            "gold": row["original_answer_letter"],
        },
        {
            "perturbation_type": "roman_numeral",
            "question": row["perturbed_question"],
            "options": parse_options_json(row["perturbed_options"]),
            "gold": row["perturbed_answer_raw"],
        },
    ]


def sanity_checks(smoke_items):
    print("=== Sanity checks ===")
    ids = smoke_items["source_id"].tolist()
    assert len(ids) == SMOKE_N, f"expected {SMOKE_N} items, got {len(ids)}"
    assert len(set(ids)) == SMOKE_N, "duplicate source_id in smoke sample"
    print(f"OK: exactly {SMOKE_N} unique underlying questions")

    for _, row in smoke_items.iterrows():
        conds = build_conditions(row)
        mcq, rom = conds[0], conds[1]

        assert list(mcq["options"].keys()) == ["A", "B", "C", "D"], \
            f"mcq option order unexpected for {row['source_id']}: {list(mcq['options'].keys())}"
        assert list(rom["options"].keys()) == ["I", "II", "III", "IV"], \
            f"roman option order unexpected for {row['source_id']}: {list(rom['options'].keys())}"

        mcq_labels, rom_labels = get_labels("mcq"), get_labels("roman_numeral")
        for i in range(4):
            assert mcq["options"][mcq_labels[i]] == rom["options"][rom_labels[i]], \
                f"option content mismatch at position {i} for {row['source_id']}"

        gold_mcq_idx = canonical_index("mcq", mcq["gold"])
        gold_rom_idx = canonical_index("roman_numeral", rom["gold"])
        assert gold_mcq_idx is not None and gold_rom_idx is not None, \
            f"unparseable gold label for {row['source_id']}"
        assert gold_mcq_idx == gold_rom_idx, \
            f"gold answer misalignment for {row['source_id']}: mcq idx={gold_mcq_idx} roman idx={gold_rom_idx}"

    print("OK: exactly one mcq + one roman_numeral representation per id, option ordering verified, "
          "gold-answer alignment verified for all 10 items")

    for name, val in (("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY")),
                       ("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))):
        assert val, f"{name} not set"
    print("OK: API keys present in environment (values not printed)")

    gi_path = ROOT / ".gitignore"
    gi_text = gi_path.read_text(encoding="utf-8")
    assert "results/behavioral/raw/" in gi_text or "results/behavioral/raw" in gi_text, \
        "results/behavioral/raw/ is not gitignored"
    assert ".env" in gi_text, ".env is not gitignored"
    print("OK: raw result directory and .env confirmed gitignored")
    print()


def main():
    import anthropic
    from google import genai

    clients = {
        "anthropic": anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "gemini": genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
    }

    df = pd.read_parquet(PILOT_PATH)
    roman = df[df["perturbation_type"] == "roman_numeral"].copy()
    roman = roman.sort_values(["source_dataset", "source_id"]).reset_index(drop=True)
    smoke_items = roman.head(SMOKE_N)

    sanity_checks(smoke_items)

    n_planned = SMOKE_N * 2 * len(MODELS)
    print(f"Planned: {SMOKE_N} questions x 2 conditions x {len(MODELS)} models = {n_planned} calls "
          f"(cache hits are free; unique models: {[m['model_id'] for m in MODELS]})\n")

    print("=== Pre-flight: one sample call per model ===")
    active_models = []
    first_row = smoke_items.iloc[0]
    first_cond = build_conditions(first_row)[0]  # mcq
    for m in MODELS:
        provider, model_id = m["provider"], m["model_id"]
        try:
            record, cached = do_call_and_cache(clients, provider, model_id,
                                                 first_row["source_dataset"], first_row["source_id"], first_cond)
            parsed = parse_answer(record["response_text"], "mcq")
            status = "CACHED-OK" if cached else "OK"
            if parsed.parsing_status != "ok":
                print(f"  {provider}/{model_id}: {status} call, but parse_status={parsed.parsing_status} "
                      f"(response={record['response_text']!r}) -> SKIPPING this model for the batch")
                continue
            print(f"  {provider}/{model_id}: {status}, parsed={parsed.parsed_label}, "
                  f"latency={record.get('latency_s')}, tokens_in={record.get('input_tokens')}, "
                  f"tokens_out={record.get('output_tokens')}")
            active_models.append(m)
        except Exception as e:
            print(f"  {provider}/{model_id}: PRE-FLIGHT FAILED ({type(e).__name__}: {e}) -> SKIPPING this model for the batch")
            log_event(event="preflight_error", provider=provider, model_id=model_id,
                       error_type=type(e).__name__, error=str(e))

    print(f"\nModels proceeding to full smoke test: {[m['model_id'] for m in active_models]}")
    skipped = [m["model_id"] for m in MODELS if m not in active_models]
    if skipped:
        print(f"Models SKIPPED (pre-flight failed, not called further): {skipped}\n")
    else:
        print()

    all_records = []
    for _, row in smoke_items.iterrows():
        source_dataset, source_id = row["source_dataset"], row["source_id"]
        for cond in build_conditions(row):
            for m in active_models:
                provider, model_id = m["provider"], m["model_id"]
                try:
                    record, cached = do_call_and_cache(clients, provider, model_id, source_dataset, source_id, cond)
                    if not cached:
                        time.sleep(0.3)
                except Exception as e:
                    log_event(event="call_error", provider=provider, model_id=model_id,
                               source_dataset=source_dataset, source_id=source_id,
                               perturbation_type=cond["perturbation_type"],
                               error_type=type(e).__name__, error=str(e))
                    print(f"ERROR {provider}/{model_id} {source_dataset}/{source_id}/{cond['perturbation_type']}: "
                          f"{type(e).__name__}: {e}")
                    continue
                all_records.append(record)

    # ---- responses.csv ----
    responses_rows = []
    for r in all_records:
        parsed = parse_answer(r["response_text"], r["perturbation_type"])
        canon = canonical_index(r["perturbation_type"], parsed.parsed_label) if parsed.parsing_status == "ok" else None
        gold_canon = canonical_index(r["perturbation_type"], r["gold_answer"])
        responses_rows.append({
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
            "estimated_cost_usd": r.get("estimated_cost_usd"),
            "sampling_config": json.dumps(r.get("sampling_config", {})),
            "prompt_version": r["prompt_version"],
            "timestamp": r["timestamp"],
        })
    responses_df = pd.DataFrame(responses_rows)
    responses_path = OUT_DIR / "phase1a_responses.csv"
    responses_df.to_csv(responses_path, index=False)

    # ---- pairs.csv ----
    pairs_rows = []
    group_iter = responses_df.groupby(["model", "question_id"]) if not responses_df.empty else []
    for (model, question_id), grp in group_iter:
        mcq_row = grp[grp["condition"] == "mcq"]
        rom_row = grp[grp["condition"] == "roman_numeral"]
        if mcq_row.empty or rom_row.empty:
            continue
        mcq_row, rom_row = mcq_row.iloc[0], rom_row.iloc[0]

        mcq_canon, rom_canon = mcq_row["canonical_answer"], rom_row["canonical_answer"]
        both_parsed = pd.notna(mcq_canon) and pd.notna(rom_canon)
        flipped = (mcq_canon != rom_canon) if both_parsed else None

        pairs_rows.append({
            "question_id": question_id,
            "model": model,
            "gold_canonical": mcq_row["gold_canonical"],
            "mcq_raw_answer": mcq_row["raw_answer"],
            "roman_raw_answer": rom_row["raw_answer"],
            "mcq_canonical": mcq_canon,
            "roman_canonical": rom_canon,
            "mcq_correct": mcq_row["correct"],
            "roman_correct": rom_row["correct"],
            "both_parsed": both_parsed,
            "flipped": flipped,
        })
    pairs_df = pd.DataFrame(pairs_rows)
    pairs_df.to_csv(OUT_DIR / "phase1a_pairs.csv", index=False)

    # ---- metrics.csv ----
    def outcome_category(row):
        if not row["both_parsed"]:
            return "unparseable"
        mc, rc = row["mcq_correct"], row["roman_correct"]
        if mc and rc:
            return "stable_correct"
        if (not mc) and (not rc):
            return "stable_incorrect" if not row["flipped"] else "incorrect_to_different_incorrect"
        if mc and not rc:
            return "correct_to_incorrect"
        return "incorrect_to_correct"

    if not pairs_df.empty:
        pairs_df["outcome_category"] = pairs_df.apply(outcome_category, axis=1)

    metrics_rows = []
    metrics_group_iter = responses_df.groupby("model") if not responses_df.empty else []
    for model, grp in metrics_group_iter:
        p = pairs_df[pairs_df["model"] == model] if not pairs_df.empty else pd.DataFrame()
        mcq_resp = grp[grp["condition"] == "mcq"]
        rom_resp = grp[grp["condition"] == "roman_numeral"]

        n_calls = len(grp)
        n_parse_fail = int((~grp["parse_success"]).sum())
        baseline_acc = mcq_resp["correct"].mean() if not mcq_resp.empty else float("nan")
        perturbed_acc = rom_resp["correct"].mean() if not rom_resp.empty else float("nan")

        n_both = int(p["both_parsed"].sum()) if not p.empty else 0
        # NOTE: `flipped`/`mcq_correct`/`roman_correct` are object-dtype columns
        # (mixed with None for unparsed rows), holding numpy.bool_ scalars.
        # `~` (invert) on an object-dtype pandas Series of numpy.bool_ does NOT
        # reliably behave as logical NOT here (confirmed: gave 0.11 instead of
        # 1.0 on an all-False column) - use explicit `== True`/`== False`
        # comparisons instead of `~`/`&` on these columns.
        both = p.loc[p["both_parsed"]]
        flip_rate = (both["flipped"] == True).mean() if n_both > 0 else float("nan")
        reacc = ((both["mcq_correct"] == True) & (both["roman_correct"] == True)).mean() if n_both > 0 else float("nan")
        recon = (both["flipped"] == False).mean() if n_both > 0 else float("nan")

        total_cost = grp["estimated_cost_usd"].sum(skipna=True)
        cost_known = grp["estimated_cost_usd"].notna().sum()

        metrics_rows.append({
            "model": model,
            "n_calls": n_calls,
            "n_parse_failures": n_parse_fail,
            "parse_failure_rate": n_parse_fail / n_calls if n_calls else float("nan"),
            "baseline_accuracy_mcq": baseline_acc,
            "perturbed_accuracy_roman_numeral": perturbed_acc,
            "accuracy_delta": (perturbed_acc - baseline_acc) if pd.notna(baseline_acc) and pd.notna(perturbed_acc) else float("nan"),
            "answer_flip_rate": flip_rate,
            "ReAcc": reacc,
            "ReCon": recon,
            "n_pairs_both_parsed": n_both,
            "estimated_cost_usd": total_cost,
            "cost_calls_with_known_tokens": int(cost_known),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "phase1a_metrics.csv", index=False)

    # ---- failures.csv ----
    if not pairs_df.empty:
        failures_df = pairs_df[pairs_df["outcome_category"] != "stable_correct"].copy()
    else:
        failures_df = pairs_df
    failures_df.to_csv(OUT_DIR / "phase1a_failures.csv", index=False)

    # ---- figure ----
    if not pairs_df.empty:
        categories = ["stable_correct", "stable_incorrect", "correct_to_incorrect",
                       "incorrect_to_correct", "incorrect_to_different_incorrect"]
        models_list = sorted(pairs_df["model"].unique())
        counts = {m: pairs_df[pairs_df["model"] == m]["outcome_category"].value_counts().reindex(categories, fill_value=0)
                  for m in models_list}

        fig, ax = plt.subplots(figsize=(11, 6))
        x = range(len(categories))
        width = 0.8 / max(len(models_list), 1)
        for i, m in enumerate(models_list):
            offsets = [xi + i * width for xi in x]
            ax.bar(offsets, counts[m].values, width=width, label=m)
        ax.set_xticks([xi + width * (len(models_list) - 1) / 2 for xi in x])
        ax.set_xticklabels([c.replace("_", " ") for c in categories], rotation=20, ha="right")
        ax.set_ylabel("Number of paired questions")
        ax.set_title("Phase 1A smoke test: mcq -> roman_numeral outcome distribution, per model")
        ax.legend()
        fig.tight_layout()
        fig_path = FIG_DIR / "phase1a_outcome_distribution.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    else:
        fig_path = None

    # ---- summary.md ----
    lines = []
    lines.append("# Phase 1A Smoke Test v2 — Summary\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"Models attempted: {[m['model_id'] for m in MODELS]}")
    lines.append(f"Models that passed pre-flight and ran: {[m['model_id'] for m in active_models]}")
    if skipped:
        lines.append(f"Models skipped after failed pre-flight: {skipped}")
    lines.append(f"\nTotal API calls made (non-cached): see logs/phase1a_smoke_v2.jsonl (event=call_success)")
    lines.append(f"Total response rows collected: {len(responses_df)}")
    lines.append(f"Total parsing failures: {int((~responses_df['parse_success']).sum()) if not responses_df.empty else 0}\n")

    lines.append("## Metrics per model\n")
    lines.append(metrics_df.to_markdown(index=False) if not metrics_df.empty else "(no data)")

    lines.append("\n## Every answer-flip case\n")
    if not pairs_df.empty:
        flips = pairs_df[pairs_df["flipped"] == True]
        if len(flips):
            lines.append(flips[["model", "question_id", "gold_canonical", "mcq_canonical", "roman_canonical",
                                  "mcq_correct", "roman_correct", "outcome_category"]].to_markdown(index=False))
        else:
            lines.append("No answer flips observed in this smoke sample.")
    else:
        lines.append("(no data)")

    total_cost = responses_df["estimated_cost_usd"].sum(skipna=True) if not responses_df.empty else 0
    lines.append(f"\n## Approximate API cost\n\nTotal estimated: **${total_cost:.4f} USD** "
                  f"(sum of per-call token-based estimates using pricing in `scripts/phase1a_smoke.py::PRICING`, "
                  f"sourced 2026-09-29; calls with missing token counts are excluded from this sum, not assumed zero-cost).")

    lines.append("\n## Suspicious pipeline behavior noted\n")
    notes = []
    if skipped:
        notes.append(f"- Pre-flight failed for: {skipped} (see logs for exact error).")
    if not responses_df.empty and (~responses_df["parse_success"]).any():
        notes.append(f"- {int((~responses_df['parse_success']).sum())} unparseable/ambiguous responses were recorded, not silently treated as wrong.")
    if not notes:
        notes.append("- None observed beyond what's reported above.")
    lines.extend(notes)

    lines.append("\n## Recommendation\n")
    n_active = len(active_models)
    if n_active == len(MODELS) and not responses_df.empty and (responses_df["parse_success"].mean() > 0.9 if not responses_df.empty else False):
        lines.append("Pipeline appears technically ready to scale, pending explicit approval — all models passed "
                      "pre-flight and parsing was reliable. Statistical conclusions still require n=200, not n=10.")
    else:
        lines.append(f"NOT fully ready to scale as-is: {n_active}/{len(MODELS)} models passed pre-flight. "
                      f"Resolve the skipped model(s) before committing budget to the full 200-question run.")

    summary_path = OUT_DIR / "phase1a_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("STOP: Phase 1A smoke test batch complete. Full 200-question cohort NOT run.")
    print("=" * 70)
    print(f"\nOutputs written to: {OUT_DIR}")
    print(f"Figure: {fig_path}")
    print(f"\nTotal estimated cost this run: ${total_cost:.4f} USD")
    print(f"\n--- {summary_path.name} ---\n")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
