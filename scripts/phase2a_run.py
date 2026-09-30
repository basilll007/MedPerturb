"""Phase 2A - single-model end-to-end framework validation.

    uv run python scripts\\phase2a_run.py --stage 1   # 1 question x 7 conditions
    uv run python scripts\\phase2a_run.py --stage 2   # 5 questions x 7
    uv run python scripts\\phase2a_run.py --stage 3   # 50 questions x 7

Every stage re-runs the semantic audit gate first and reuses any cached
result produced under the identical configuration.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medperturb.adapters.gemini import GeminiAdapter  # noqa: E402
from medperturb.evaluation import metrics as M  # noqa: E402
from medperturb.evaluation.cache import CACHE_SCHEMA, ResponseCache  # noqa: E402
from medperturb.evaluation.qc import independent_recompute, structural_checks  # noqa: E402
from medperturb.evaluation.runner import run  # noqa: E402
from medperturb.evaluation.semantic import EVALUATOR_VERSION  # noqa: E402
from medperturb.parsing.canonical import PARSER_VERSION  # noqa: E402
from medperturb.perturbations.audit import audit_question  # noqa: E402
from medperturb.perturbations.registry import CONDITIONS, REGISTRY, build_examples  # noqa: E402
from medperturb.prompts.templates import PROMPT_VERSION  # noqa: E402

PILOT = ROOT / "data" / "processed" / "pilot_200.parquet"
COHORT = ROOT / "data" / "processed" / "phase2a_50_manifest.csv"
RAW_ROOT = ROOT / "results" / "behavioral" / "raw_phase2a"
OUT_ROOT = ROOT / "results" / "behavioral" / "phase2a"

MODEL = "gemini-3.8-flash"
THINKING_LEVEL, MAX_OUTPUT_TOKENS = "low", 1536
STAGES = {1: (1, 0.05), 2: (5, 0.25), 3: (50, 1.50)}  # n_questions, max NEW spend USD
MAX_TRUNCATION_RATE_STAGE3 = 0.02


class Stop(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def preflight():
    for probe in ("results/behavioral/raw_phase2a/probe.json", ".env"):
        r = subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT)
        if r.returncode != 0:
            raise Stop(f"{probe} is NOT gitignored - refusing to run")
    tracked = git("ls-files", "results/behavioral/raw_phase2a", ".env") or ""
    if tracked:
        raise Stop(f"raw cache or .env is tracked by git: {tracked}")
    if not os.environ.get("GEMINI_API_KEY"):
        raise Stop("GEMINI_API_KEY not set")
    print("Preflight OK: raw cache + .env gitignored and untracked; API key present (value not printed).")


def load_cohort():
    man = pd.read_csv(COHORT).sort_values("stage_order")
    assert man["question_id"].is_unique and len(man) == 50
    pilot = pd.read_parquet(PILOT)
    pilot_ids = set(pilot["source_dataset"] + "::" + pilot["source_id"])
    assert set(man["question_id"]) <= pilot_ids, "cohort contains ids outside pilot_200"
    return man, pilot


def semantic_audit_gate(man, pilot):
    rows, contradictions, all_examples = [], [], {}
    for _, m in man.iterrows():
        sub = pilot[(pilot["source_dataset"] == m["source_dataset"]) & (pilot["source_id"] == m["source_id"])]
        ex = build_examples(sub)  # raises on any missing representation - never silently dropped
        r, c = audit_question(ex, sub)
        rows += r
        contradictions += c
        all_examples[m["question_id"]] = ex
    audit = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_ROOT / "semantic_audit.csv", index=False)
    write_audit_summary(audit, contradictions)
    if contradictions:
        raise Stop(f"semantic audit found {len(contradictions)} contradictions - see semantic_audit.csv")
    print(f"Semantic audit OK: {len(audit)} (question, condition) rows, 0 contradictions.")
    return all_examples, audit


def write_audit_summary(audit, contradictions):
    L = ["# Phase 2A — Pre-run semantic audit\n",
         f"Rows audited: {len(audit)} (50 questions x 7 representations). Contradictions: **{len(contradictions)}**.\n",
         "Each row is checked against the registry's claimed taxonomy AND ReMedQA's own dataset prompt "
         "(displayed options, label style, answer format, task inversion).\n",
         "| condition | category | q text changed | gold content changed | gold position changed | observed display | observed format | expectation met |",
         "|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        g = audit[audit["condition"] == cond]
        pos = g["gold_position_changed"]
        pos_s = "n/a (no options shown)" if pos.isna().all() else f"{int((pos == True).sum())}/{len(g)}"  # noqa: E712
        L.append(f"| {cond} | {REGISTRY[cond].category} | {int(g['question_text_changed'].sum())}/{len(g)} | "
                 f"{int((g['gold_content_changed'] == True).sum())}/{len(g)} | {pos_s} | "  # noqa: E712
                 f"{', '.join(sorted(g['observed_display'].unique()))} | {', '.join(sorted(g['observed_response_format'].unique()))} | "
                 f"{int(g['expectation_met'].sum())}/{len(g)} |")
    L += ["", "## Operational meaning, as verified", ""]
    for cond in CONDITIONS:
        s = REGISTRY[cond]
        L.append(f"- **{cond}** — {s.category}. {s.description} {s.notes}".rstrip())
    L += ["", f"Option counts observed: {audit['option_count'].value_counts().to_dict()}. "
              f"Questions with duplicate option texts (would break text-based semantic identity): "
              f"{int(audit['duplicate_option_texts'].sum())}.", ""]
    if contradictions:
        L += ["## Contradictions", ""] + [f"- {c}" for c in contradictions]
    (OUT_ROOT / "semantic_audit_summary.md").write_text("\n".join(L), encoding="utf-8")


def transitions_table(paired):
    rows = []
    for cond in [c for c in CONDITIONS if c != "mcq"]:
        p = paired[(paired["condition"] == cond) & (paired["both_evaluable"] == True)]  # noqa: E712
        for t, k in p["transition"].value_counts().items():
            rows.append({"condition": cond, "kind": "correctness_transition", "label": t, "count": int(k)})
        for t, k in p["adaptation"].dropna().value_counts().items():
            rows.append({"condition": cond, "kind": "adaptation", "label": t, "count": int(k)})
        n_excl = int((paired["condition"] == cond).sum()) - len(p)
        rows.append({"condition": cond, "kind": "excluded_not_both_evaluable", "label": "excluded", "count": n_excl})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    args = ap.parse_args()
    n_q, max_cost = STAGES[args.stage]

    load_dotenv(ROOT / ".env")
    preflight()
    man, pilot = load_cohort()
    all_examples, audit = semantic_audit_gate(man, pilot)

    stage_ids = man["question_id"].tolist()[:n_q]
    examples = [all_examples[q][c] for q in stage_ids for c in CONDITIONS]

    adapter = GeminiAdapter(MODEL, os.environ["GEMINI_API_KEY"], THINKING_LEVEL, MAX_OUTPUT_TOKENS)
    config_hash = adapter.config_hash(PROMPT_VERSION, CACHE_SCHEMA)
    cache = ResponseCache(RAW_ROOT, config_hash, [os.environ.get("GEMINI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY")])
    print(f"Stage {args.stage}: {n_q} question(s) x {len(CONDITIONS)} = {len(examples)} evaluations; "
          f"config_hash={config_hash[:16]}; new-spend ceiling ${max_cost}")

    result = run(examples, adapter, cache, max_cost)
    print(f"Run: new calls={result.n_new_calls} cached={result.n_cached} failed={result.n_failed} "
          f"retries={result.n_retries} new spend=${result.new_cost_usd:.4f}")

    resp = M.build_responses(result.records)
    paired = M.build_paired(resp)
    metrics = M.metrics_by_perturbation(resp, paired)
    group = M.group_consistency(resp)
    failures = M.build_failures(resp)
    trans = transitions_table(paired)

    errs = structural_checks(examples, result.records, resp, n_q, config_hash)
    errs += independent_recompute(resp, metrics, paired, group)

    out = OUT_ROOT if args.stage == 3 else OUT_ROOT / f"stage{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    resp.to_csv(out / "responses.csv", index=False)
    paired.to_csv(out / "paired_results.csv", index=False)
    metrics.to_csv(out / "metrics_by_perturbation.csv", index=False)
    trans.to_csv(out / "transitions.csv", index=False)
    failures.to_csv(out / "failures.csv", index=False)
    (ROOT / "results" / "behavioral" / "phase2a").mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT_ROOT / "cohort_manifest.csv", index=False)

    n_trunc = int(resp["truncated"].sum())
    manifest = {
        "experiment_id": f"phase2a_stage{args.stage}_{config_hash[:10]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "dataset": {"path": str(PILOT.relative_to(ROOT)), "sha256": sha256_file(PILOT)},
        "cohort": {"path": str(COHORT.relative_to(ROOT)), "sha256_ids": hashlib.sha256("\n".join(man["question_id"]).encode()).hexdigest(),
                   "stage_question_ids": stage_ids},
        "provider": adapter.provider, "model": MODEL,
        "model_versions_observed": sorted({v for v in resp["model_version"].dropna().unique()}),
        "perturbations": {c: REGISTRY[c].category for c in CONDITIONS},
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256_distinct": int(resp["prompt"].map(lambda p: hashlib.sha256(p.encode()).hexdigest()).nunique()),
        "inference_config": adapter.inference_config(), "config_hash": config_hash,
        "parser_version": PARSER_VERSION, "evaluator_version": EVALUATOR_VERSION, "cache_schema": CACHE_SCHEMA,
        "n_expected": len(examples), "n_new_calls": result.n_new_calls, "n_cached": result.n_cached,
        "n_failed": result.n_failed, "n_retries": result.n_retries, "n_truncated": n_trunc,
        "new_cost_usd": round(result.new_cost_usd, 6), "total_cost_usd_all_records": round(float(resp["cost_usd"].sum()), 6),
        "group_consistency": group, "qc_errors": errs,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if args.stage == 1:
        for (ex, prompt, r, cached), (_, row) in zip(result.records, resp.iterrows()):
            print(f"\n--- {ex.condition} ---\n{prompt}\n>>> raw={r.raw_response!r} finish={r.finish_reason} "
                  f"think={r.thinking_tokens} out={r.output_tokens} | parsed={row['normalized_answer']} "
                  f"eval={row['match_method']} correct={row['correct']} gold={row['gold_label']}/{row['gold_option_text']!r}")

    print("\n" + metrics[["condition", "N", "n_parsed", "n_evaluable", "n_correct", "accuracy", "n_truncated",
                          "mean_thinking_tokens"]].to_string(index=False))
    print(f"\nJoint consistency: {group}")
    print(f"Failures by type: {failures['failure_type'].value_counts().to_dict()}")

    stops = []
    if errs:
        stops.append(f"QC/independent recompute errors: {errs}")
    if result.n_failed:
        stops.append(f"{result.n_failed} provider failures: {result.failures}")
    if args.stage < 3 and n_trunc:
        stops.append(f"{n_trunc} truncations in stage {args.stage}")
    if args.stage == 3 and n_trunc / len(resp) > MAX_TRUNCATION_RATE_STAGE3:
        stops.append(f"truncation rate {n_trunc / len(resp):.1%} > {MAX_TRUNCATION_RATE_STAGE3:.0%}")
    if args.stage < 3 and (~resp["parse_success"]).any():
        stops.append(f"parse failures in stage {args.stage}: {resp.loc[~resp['parse_success'], ['condition', 'parse_failure_reason']].values.tolist()}")
    print(f"\nOutputs -> {out}")
    if stops:
        print("\n!!! STOP CONDITIONS !!!\n" + "\n".join(f"- {s}" for s in stops))
        sys.exit(2)
    print(f"\nStage {args.stage}: all assertions passed.")


if __name__ == "__main__":
    try:
        main()
    except Stop as e:
        print(f"\n!!! STOP: {e}")
        sys.exit(2)
