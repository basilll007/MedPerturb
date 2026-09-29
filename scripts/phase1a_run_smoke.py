"""Phase 1A - CONTROLLED BEHAVIORAL PILOT - SMOKE TEST ONLY.

Runs exactly 10 source questions x 2 conditions (mcq baseline, roman_numeral)
x 2 models (Claude, Gemini) = 40 API calls total. This script intentionally
does NOT accept a flag to run the full 200-question cohort - scaling up is a
separate, explicitly-approved step (see PHASE1A_NOTES.md).

Run: uv run python scripts\\phase1a_run_smoke.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1a_prompts import PROMPT_VERSION, build_prompt, parse_options_json, get_labels  # noqa: E402
from phase1a_parse import parse_answer  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "data" / "processed" / "pilot_200.parquet"
RAW_DIR = ROOT / "results" / "behavioral" / "raw"
PARSED_DIR = ROOT / "results" / "behavioral" / "parsed"
LOG_DIR = ROOT / "logs"

SMOKE_N = 10
TEMPERATURE = 0
MAX_TOKENS = 16

MODELS = [
    {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
    {"provider": "gemini", "model_id": "gemini-2.5-flash-lite"},
]

for d in (RAW_DIR, PARSED_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "phase1a_smoke.jsonl"


def log_event(**kwargs):
    kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(kwargs) + "\n")


def to_serializable(obj):
    for attr in ("model_dump",):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)(mode="json")
            except TypeError:
                try:
                    return getattr(obj, attr)()
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


def raw_path(provider, model_id, source_dataset, source_id, perturbation_type):
    safe_model = model_id.replace("/", "_")
    d = RAW_DIR / provider / safe_model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{source_dataset}_{source_id}_{perturbation_type}.json"


def call_anthropic(client, model_id, prompt, labels):
    # NOTE: as of anthropic SDK 1.9.0 (live API, checked 2026-09-29), the
    # Messages API no longer accepts `temperature` at all (confirmed via
    # inspect.signature - the parameter does not exist in this SDK version).
    # The closest available "lowest-variance" control is `effort=low`.
    # This is recorded explicitly here and in results, per the Phase 1A
    # requirement to never silently substitute/omit a specified setting.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": labels}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": schema},
        },
    )
    text = ""
    if resp.content and len(resp.content) > 0:
        text = getattr(resp.content[0], "text", "") or ""
    # Structured output returns a JSON object; extract the "answer" field so
    # downstream parsing sees a bare label like the Gemini free-text path does.
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict) and "answer" in parsed_json:
            text = parsed_json["answer"]
    except (json.JSONDecodeError, TypeError):
        pass  # leave text as-is; parse_answer will mark it unparseable/ambiguous, not silently guess
    return text, to_serializable(resp)


def call_gemini(client, model_id, prompt):
    from google.genai import types

    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    text = resp.text or "" if hasattr(resp, "text") else ""
    return text, to_serializable(resp)


def main():
    import anthropic
    from google import genai
    import os

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    df = pd.read_parquet(PILOT_PATH)
    roman = df[df["perturbation_type"] == "roman_numeral"].copy()
    roman = roman.sort_values(["source_dataset", "source_id"]).reset_index(drop=True)
    smoke_items = roman.head(SMOKE_N)

    print(f"Smoke test: {len(smoke_items)} source questions x 2 conditions x {len(MODELS)} models "
          f"= {len(smoke_items) * 2 * len(MODELS)} planned API calls (cache-skipped calls don't cost anything).")
    assert len(smoke_items) == SMOKE_N, "Pilot cohort smaller than expected smoke-test size"

    total_calls_made = 0
    total_cache_hits = 0
    parsed_rows = []

    for _, row in smoke_items.iterrows():
        source_dataset = row["source_dataset"]
        source_id = row["source_id"]

        conditions = [
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

        for cond in conditions:
            prompt = build_prompt(cond["perturbation_type"], cond["question"], cond["options"])

            for m in MODELS:
                provider, model_id = m["provider"], m["model_id"]
                target = raw_path(provider, model_id, source_dataset, source_id, cond["perturbation_type"])

                if target.exists():
                    total_cache_hits += 1
                    log_event(event="cache_hit", provider=provider, model_id=model_id,
                               source_dataset=source_dataset, source_id=source_id,
                               perturbation_type=cond["perturbation_type"])
                else:
                    log_event(event="call_start", provider=provider, model_id=model_id,
                               source_dataset=source_dataset, source_id=source_id,
                               perturbation_type=cond["perturbation_type"])
                    try:
                        if provider == "anthropic":
                            text, raw = call_anthropic(anthropic_client, model_id, prompt,
                                                        get_labels(cond["perturbation_type"]))
                            sampling_config = {"temperature": "unsupported_by_api", "effort": "low"}
                        else:
                            text, raw = call_gemini(gemini_client, model_id, prompt)
                            sampling_config = {"temperature": TEMPERATURE}

                        record = {
                            "provider": provider,
                            "model_id": model_id,
                            "source_dataset": source_dataset,
                            "source_id": source_id,
                            "perturbation_type": cond["perturbation_type"],
                            "gold_answer": cond["gold"],
                            "prompt_version": PROMPT_VERSION,
                            "prompt_text": prompt,
                            "sampling_config": sampling_config,
                            "max_tokens": MAX_TOKENS,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "response_text": text,
                            "raw_response": raw,
                        }
                        with open(target, "w", encoding="utf-8") as f:
                            json.dump(record, f, indent=2, default=str)

                        total_calls_made += 1
                        log_event(event="call_success", provider=provider, model_id=model_id,
                                   source_dataset=source_dataset, source_id=source_id,
                                   perturbation_type=cond["perturbation_type"],
                                   response_text=text)
                        time.sleep(0.3)
                    except Exception as e:
                        log_event(event="call_error", provider=provider, model_id=model_id,
                                   source_dataset=source_dataset, source_id=source_id,
                                   perturbation_type=cond["perturbation_type"],
                                   error_type=type(e).__name__, error=str(e))
                        print(f"ERROR calling {provider}/{model_id} on {source_dataset}/{source_id}/"
                              f"{cond['perturbation_type']}: {type(e).__name__}: {e}")
                        continue

                # Parse (whether just-called or cache-hit) from the persisted raw file
                with open(target, "r", encoding="utf-8") as f:
                    record = json.load(f)

                parsed = parse_answer(record["response_text"], cond["perturbation_type"])
                correctness = None
                if parsed.parsing_status == "ok":
                    correctness = (parsed.parsed_label.upper() == str(record["gold_answer"]).upper())

                parsed_rows.append({
                    "provider": provider,
                    "model_id": model_id,
                    "source_dataset": source_dataset,
                    "source_id": source_id,
                    "perturbation_type": cond["perturbation_type"],
                    "gold_answer": record["gold_answer"],
                    "raw_response_text": record["response_text"],
                    "parsed_prediction": parsed.parsed_label,
                    "parsing_status": parsed.parsing_status,
                    "correct": correctness,
                    "prompt_version": record["prompt_version"],
                    # backward-compatible with raw files cached before sampling_config existed
                    "sampling_config": json.dumps(record.get("sampling_config", {"temperature": record.get("temperature")})),
                    "timestamp": record["timestamp"],
                })

    parsed_df = pd.DataFrame(parsed_rows)
    out_path = PARSED_DIR / "phase1a_smoke_parsed.csv"
    parsed_df.to_csv(out_path, index=False)

    print(f"\nAPI calls made this run: {total_calls_made}")
    print(f"Cache hits (skipped, restart-safe): {total_cache_hits}")
    print(f"Parsed results written to: {out_path}")
    print("\n=== Per model x condition summary ===")
    summary = parsed_df.groupby(["provider", "model_id", "perturbation_type"]).agg(
        n=("correct", "size"),
        parse_ok_rate=("parsing_status", lambda s: (s == "ok").mean()),
        accuracy_among_parsed=("correct", lambda s: s.dropna().mean() if s.notna().any() else float("nan")),
    ).reset_index()
    print(summary.to_string(index=False))

    print("\n=== Paired behavioral transitions (mcq -> roman_numeral), per model ===")
    pivot = parsed_df.pivot_table(
        index=["provider", "model_id", "source_dataset", "source_id"],
        columns="perturbation_type",
        values="correct",
        aggfunc="first",
    ).reset_index()

    for (provider, model_id), grp in pivot.groupby(["provider", "model_id"]):
        valid = grp.dropna(subset=["mcq", "roman_numeral"])
        n_valid = len(valid)
        n_excluded = len(grp) - n_valid
        if n_valid == 0:
            print(f"{provider}/{model_id}: no fully-parsed pairs (n_excluded_unparseable={n_excluded})")
            continue
        cc = ((valid["mcq"] == True) & (valid["roman_numeral"] == True)).sum()
        cw = ((valid["mcq"] == True) & (valid["roman_numeral"] == False)).sum()
        wc = ((valid["mcq"] == False) & (valid["roman_numeral"] == True)).sum()
        ww = ((valid["mcq"] == False) & (valid["roman_numeral"] == False)).sum()
        print(f"{provider}/{model_id}: n_valid_pairs={n_valid} (excluded unparseable={n_excluded}) "
              f"| C->C={cc} C->W={cw} W->C={wc} W->W={ww}")

    remaining_originals = 200 - SMOKE_N
    remaining_requests = remaining_originals * 2 * len(MODELS)
    print(f"\nEstimated remaining API requests for full Phase 1A (200 originals, mcq+roman_numeral, "
          f"{len(MODELS)} models): {remaining_requests} "
          f"(plus this smoke test's {SMOKE_N * 2 * len(MODELS)} = {SMOKE_N * 2 * len(MODELS) + remaining_requests} total for the full cohort)")
    print("\nSTOP: smoke test complete. Do not scale to the remaining 190 questions without explicit approval.")


if __name__ == "__main__":
    main()
