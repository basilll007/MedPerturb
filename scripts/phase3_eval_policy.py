"""Phase 3 — Generalized Policy Evaluation Runner.

Evaluates any model/adapter policy (Base, Correctness, Neuro-Symbolic)
across the MedPerturb perturbations using the exact same evaluation framework,
metrics, caching, and independent QC.

Usage:
    uv run --no-sync python scripts/phase3_eval_policy.py --policy_name correctness --adapter_path results/phase3/checkpoints/policy_correctness --stage diagnostic
    uv run --no-sync python scripts/phase3_eval_policy.py --policy_name neurosymbolic --adapter_path results/phase3/checkpoints/policy_neurosymbolic --stage diagnostic
    uv run --no-sync python scripts/phase3_eval_policy.py --policy_name neurosymbolic --adapter_path results/phase3/checkpoints/policy_neurosymbolic --stage final_test
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
    for cond, g in paired.groupby("condition"):
        t = g[g["both_evaluable"]]["transition"].value_counts().to_dict()
        n = int(g["both_evaluable"].sum())
        rows.append({
            "condition": cond,
            "both_evaluable": n,
            "corr_to_corr": t.get("corr_to_corr", 0),
            "corr_to_inc": t.get("corr_to_inc", 0),
            "inc_to_corr": t.get("inc_to_corr", 0),
            "inc_to_inc": t.get("inc_to_inc", 0),
        })
    return pd.DataFrame(rows)


def load_cohort(stage: str) -> tuple[list[str], pd.DataFrame]:
    man = pd.read_csv(SPLIT_MANIFEST)
    pairs = pd.read_parquet(PAIRS_PARQUET)

    if stage in ("1", "stage_1"):
        qids = [man[man["split"] == "diagnostic"]["question_id"].iloc[0]]
    elif stage in ("2", "stage_2"):
        qids = man[man["split"] == "diagnostic"]["question_id"].iloc[:5].tolist()
    elif stage in ("diagnostic", "val", "final_test", "train"):
        qids = man[man["split"] == stage]["question_id"].tolist()
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


def evaluate_policy(
    policy_name: str,
    stage: str,
    adapter_path: Path | str | None = None,
    max_questions: int | None = None,
) -> Path:
    raw_root = ROOT / "results" / "behavioral" / f"raw_phase3_{policy_name}"
    out_dir = ROOT / "results" / "phase3" / "evaluations" / policy_name / f"stage_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Policy Evaluation: {policy_name} ({stage}) ===")
    if adapter_path:
        print(f"    Adapter Path: {adapter_path}")
    print(f"    Output Directory: {out_dir}")

    stage_qids, pairs_df = load_cohort(stage)
    if max_questions is not None:
        stage_qids = stage_qids[:max_questions]
    n_q = len(stage_qids)

    all_examples, audit_df = semantic_audit_gate(stage_qids, pairs_df)
    examples = [all_examples[q][c] for q in stage_qids for c in CONDITIONS]

    print(f"    Questions: {n_q}, Evaluations: {len(examples)} ({n_q} x {len(CONDITIONS)})")

    adapter = QwenAdapter(model_id=MODEL_ID, adapter_path=adapter_path)
    config_hash = adapter.config_hash(PROMPT_VERSION, CACHE_SCHEMA)
    cache = ResponseCache(raw_root, config_hash, secret_values=[])

    print(f"    Config hash: {config_hash[:16]}")
    t0 = time.time()
    result = run(examples, adapter, cache, max_new_cost_usd=1000.0, pause_s=0.0)
    total_time = time.time() - t0
    adapter.unload()

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
        "experiment_id": f"phase3_{policy_name}_{stage}_{config_hash[:10]}",
        "policy_name": policy_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "stage": stage,
        "model_id": MODEL_ID,
        "inference_config": adapter.inference_config(),
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "config_hash": config_hash,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_clean": git("status", "--porcelain") == "",
        "random_seed": 42,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": n_q,
        "n_expected_calls": len(examples),
        "n_new_calls": result.n_new_calls,
        "n_cached": result.n_cached,
        "n_failed": result.n_failed,
        "total_runtime_s": round(total_time, 2),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "pairs_parquet_sha256": sha256_file(PAIRS_PARQUET),
        "qc_errors": errs,
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + metrics[["condition", "N", "n_parsed", "n_evaluable", "n_correct", "accuracy"]].to_string())
    print(f"\nJoint consistency: {group}")
    print(f"Failures by type: {failures['failure_type'].value_counts().to_dict()}")

    if errs:
        raise RuntimeError(f"QC failed with errors: {errs}")

    print(f"\n[OK] Evaluation for {policy_name} ({stage}) completed successfully. Saved to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="MedPerturb Phase 3 Policy Evaluation.")
    parser.add_argument("--policy_name", type=str, required=True, help="e.g. base, correctness, neurosymbolic")
    parser.add_argument("--stage", type=str, required=True, choices=["1", "2", "diagnostic", "val", "final_test"])
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA checkpoint directory")
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    evaluate_policy(
        policy_name=args.policy_name,
        stage=args.stage,
        adapter_path=args.adapter_path,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
