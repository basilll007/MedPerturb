"""Phase 3 — Qwen3-4B Baseline Evaluation Runner.

Evaluates base Qwen3-4B across the MedPerturb perturbations using the existing
evaluation framework, semantic evaluators, metrics, caching, and independent QC.

Stages:
    uv run --no-sync python scripts/phase3_eval_baseline.py --stage 1
    uv run --no-sync python scripts/phase3_eval_baseline.py --stage 2
    uv run --no-sync python scripts/phase3_eval_baseline.py --stage diagnostic
    uv run --no-sync python scripts/phase3_eval_baseline.py --stage final_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medperturb.adapters.qwen import QwenAdapter
from medperturb.evaluation import metrics as M
from medperturb.evaluation.cache import CACHE_SCHEMA, ResponseCache
from medperturb.evaluation.qc import independent_recompute, structural_checks
from medperturb.evaluation.runner import run
from medperturb.evaluation.semantic import EVALUATOR_VERSION
from medperturb.parsing.canonical import PARSER_VERSION
from medperturb.perturbations.audit import audit_question
from medperturb.perturbations.registry import CONDITIONS, REGISTRY, build_examples
from medperturb.prompts.templates import PROMPT_VERSION

SPLIT_MANIFEST = ROOT / "data" / "processed" / "phase3" / "split_manifest.csv"
PAIRS_PARQUET = ROOT / "data" / "processed" / "remedqa_pairs.parquet"
RAW_ROOT = ROOT / "results" / "behavioral" / "raw_phase3_baseline"
OUT_ROOT = ROOT / "results" / "phase3" / "baseline"

MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def transitions_table(paired: pd.DataFrame) -> pd.DataFrame:
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


def load_cohort(stage: str) -> tuple[list[str], pd.DataFrame]:
    man = pd.read_csv(SPLIT_MANIFEST)
    pairs = pd.read_parquet(PAIRS_PARQUET)

    if stage == "1":
        qids = [man[man["split"] == "diagnostic"]["question_id"].iloc[0]]
    elif stage == "2":
        qids = man[man["split"] == "diagnostic"]["question_id"].iloc[:5].tolist()
    elif stage == "diagnostic":
        qids = man[man["split"] == "diagnostic"]["question_id"].tolist()
    elif stage == "final_test":
        qids = man[man["split"] == "final_test"]["question_id"].tolist()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return qids, pairs


def semantic_audit_gate(qids: list[str], pairs: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows, contradictions, all_examples = [], [], {}
    for qid in qids:
        src, sid = qid.split("::")
        sub = pairs[(pairs["source_dataset"] == src) & (pairs["source_id"] == sid)]
        ex = build_examples(sub)
        r, c = audit_question(ex, sub)
        rows += r
        contradictions += c
        all_examples[qid] = ex

    audit_df = pd.DataFrame(rows)
    if contradictions:
        raise RuntimeError(f"Semantic audit found {len(contradictions)} contradictions")
    return all_examples, audit_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-4B Baseline Evaluation.")
    parser.add_argument("--stage", type=str, required=True, choices=["1", "2", "diagnostic", "final_test"])
    args = parser.parse_args()

    stage_name = f"stage_{args.stage}"
    out_dir = OUT_ROOT / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading cohort for stage: {args.stage}...")
    stage_qids, pairs_df = load_cohort(args.stage)
    n_q = len(stage_qids)

    all_examples, audit_df = semantic_audit_gate(stage_qids, pairs_df)
    examples = [all_examples[q][c] for q in stage_qids for c in CONDITIONS]

    print(f"    Questions: {n_q}, Evaluations: {len(examples)} ({n_q} x {len(CONDITIONS)})")

    print("[*] Initializing QwenAdapter...")
    adapter = QwenAdapter(model_id=MODEL_ID)
    config_hash = adapter.config_hash(PROMPT_VERSION, CACHE_SCHEMA)
    cache = ResponseCache(RAW_ROOT, config_hash, secret_values=[])

    print(f"    Config hash: {config_hash[:16]}")
    t0 = time.time()
    result = run(examples, adapter, cache, max_new_cost_usd=1000.0, pause_s=0.0)
    total_time = time.time() - t0

    print(f"[+] Run complete: new_calls={result.n_new_calls}, cached={result.n_cached}, failed={result.n_failed}, retries={result.n_retries}")

    resp = M.build_responses(result.records)
    paired = M.build_paired(resp)
    metrics = M.metrics_by_perturbation(resp, paired)
    group = M.group_consistency(resp)
    failures = M.build_failures(resp)
    trans = transitions_table(paired)

    errs = structural_checks(examples, result.records, resp, n_q, config_hash)
    errs += independent_recompute(resp, metrics, paired, group)

    resp.to_csv(out_dir / "responses.csv", index=False)
    paired.to_csv(out_dir / "paired_results.csv", index=False)
    metrics.to_csv(out_dir / "metrics_by_perturbation.csv", index=False)
    trans.to_csv(out_dir / "transitions.csv", index=False)
    failures.to_csv(out_dir / "failures.csv", index=False)

    manifest = {
        "experiment_id": f"phase3_baseline_{stage_name}_{config_hash[:10]}",
        "stage": args.stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "model_id": MODEL_ID,
        "n_questions": n_q,
        "n_expected": len(examples),
        "n_new_calls": result.n_new_calls,
        "n_cached": result.n_cached,
        "n_failed": result.n_failed,
        "config_hash": config_hash,
        "runtime_seconds": round(total_time, 2),
        "qc_errors": errs,
        "group_consistency": group,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n" + metrics[["condition", "N", "n_parsed", "n_evaluable", "n_correct", "accuracy"]].to_string(index=False))
    print(f"\nJoint consistency: {group}")
    print(f"Failures by type: {failures['failure_type'].value_counts().to_dict()}")

    if errs:
        print(f"\n[!] STOP: QC errors encountered: {errs}")
        sys.exit(2)

    print(f"\n[OK] Stage {args.stage} baseline evaluation finished successfully. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
