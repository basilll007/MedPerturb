"""Model-agnostic experiment runner: prompt -> cache check -> adapter call ->
cache write, with resume, duplicate-call protection, and a hard cost ceiling."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from medperturb.adapters.base import ModelAdapter
from medperturb.evaluation.cache import ResponseCache
from medperturb.perturbations.registry import output_spec_for
from medperturb.prompts.templates import PROMPT_VERSION, build_prompt, prompt_sha256
from medperturb.schemas.response import Example, ModelResponse


class CostCeilingExceeded(RuntimeError):
    pass


@dataclass
class RunResult:
    records: list = field(default_factory=list)  # (Example, prompt, ModelResponse, cached: bool)
    n_expected: int = 0
    n_new_calls: int = 0
    n_cached: int = 0
    n_failed: int = 0
    n_retries: int = 0
    new_cost_usd: float = 0.0
    failures: list = field(default_factory=list)


def run(examples: list[Example], adapter: ModelAdapter, cache: ResponseCache, max_new_cost_usd: float,
        log=print, max_retries: int = 1, pause_s: float = 0.2) -> RunResult:
    res = RunResult(n_expected=len(examples))
    seen = set()
    for ex in examples:
        key = (ex.question_id, ex.condition)
        if key in seen:
            raise AssertionError(f"duplicate evaluation requested: {key}")
        seen.add(key)

        prompt = build_prompt(ex)
        sha = prompt_sha256(prompt)
        cached = cache.get(ex.question_id, ex.condition, sha)
        if cached is not None:
            res.n_cached += 1
            res.records.append((ex, prompt, cached, True))
            continue

        if res.new_cost_usd >= max_new_cost_usd:
            raise CostCeilingExceeded(f"new spend ${res.new_cost_usd:.4f} reached ceiling ${max_new_cost_usd}")

        resp, attempt, last_err = None, 0, None
        while attempt <= max_retries and resp is None:
            attempt += 1
            try:
                resp = adapter.generate(ex, prompt, output_spec_for(ex))
            except Exception as e:  # transport/provider error - not cached, retried once
                last_err = f"{type(e).__name__}: {e}"
                log(f"  provider error ({key}, attempt {attempt}): {last_err[:200]}")
                if attempt <= max_retries:
                    res.n_retries += 1
                    time.sleep(2.0)

        if resp is None:
            res.n_failed += 1
            res.failures.append((key, last_err))
            failed = ModelResponse(adapter.provider, adapter.model_id, None, ex.question_id, ex.condition, None,
                                   None, None, None, None, None, datetime.now(timezone.utc).isoformat(), None,
                                   error=last_err)
            res.records.append((ex, prompt, failed, False))
            continue

        cache.put(resp, prompt, sha, PROMPT_VERSION)
        res.n_new_calls += 1
        res.new_cost_usd += resp.cost_usd or 0.0
        res.records.append((ex, prompt, resp, False))
        time.sleep(pause_s)
    return res
