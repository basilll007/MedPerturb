# Phase 1A — Controlled Behavioral Pilot — Smoke Test Report

Date: 2026-09-29
Scope: 10 source questions × 2 conditions (mcq baseline, roman_numeral) × 2 models = 40 planned API calls.
Status: **PARTIAL — STOPPED per protocol pending approval, and blocked on Anthropic account usage cap.**

## Exact model identifiers used

Discovered live via each provider's models-list endpoint (metadata-only call, not guessed):

- **Claude**: `claude-haiku-4-5-20251001` (Anthropic SDK `anthropic==1.9.0`)
- **Gemini**: `gemini-2.5-flash-lite` (SDK `google-genai==2.25.0`)

Both are the cheapest stable (non-preview, non-"latest"-alias) tier in their family, chosen for cost control per explicit instruction not to overspend. Dated/pinned snapshots were preferred over rolling aliases (`gemini-flash-lite-latest`, etc.) for reproducibility — a rolling alias would silently point at a different model on a future rerun.

**This tier choice needs your confirmation before scaling to the full 200-question cohort.** Cheap/fast models are appropriate for smoke-testing pipeline mechanics; whether the *actual* Phase 1A science should use these same cheap tiers or a flagship tier (Opus/Sonnet, Gemini Pro) is a separate methodological decision affecting how the results should be framed (budget-model robustness vs. frontier-model robustness).

## Real API behavior discovered (not assumed)

- **The Anthropic Messages API (as of this SDK/account, checked 2026-09-29) no longer accepts a `temperature` parameter at all** — confirmed via `inspect.signature`, not just a guess from a doc mismatch. Sampling control has moved to an `effort` parameter (`low`/`medium`/`high`/`xhigh`/`max`) plus native JSON-schema structured output. Used `effort="low"` as the closest available "lowest-variance" setting, and used a strict JSON-schema enum (`{"answer": {"enum": [...]}}`) to constrain Claude's output to a valid label — a stronger determinism-of-format guarantee than free-text + regex parsing.
- Gemini's `generateContent` still accepts `temperature`; used `temperature=0` as specified.
- **This asymmetry (temperature supported on one provider, not the other) is recorded per-row** in `sampling_config` (JSON string) in the parsed results, not silently normalized away.

## Results

### Gemini (`gemini-2.5-flash-lite`) — COMPLETE, 20/20 calls succeeded
- Parse success rate: **100%** (10/10 mcq, 10/10 roman_numeral) — the JSON-schema-free plain-text prompt + regex parser needed zero fallback handling on this sample.
- Accuracy among parsed: mcq 60% (6/10), roman_numeral 60% (6/10).
- Paired transitions (mcq → roman_numeral): **C→C=6, C→W=0, W→C=0, W→W=4** (n=10 valid pairs, 0 excluded as unparseable).
- Caveat: n=10 is a pipeline-validation sample only, not remotely powered for any statistical claim. Zero flips observed here should not be read as "roman_numeral causes no failures" — the full 200-question run is what determines that.

### Claude (`claude-haiku-4-5-20251001`) — BLOCKED, 0/20 calls succeeded
- Every call returned `400 invalid_request_error: "You have reached your specified API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."`
- This is an account-level usage cap, not a per-request or per-model issue, and not something a retry or backoff resolves — confirmed identical on a second full rerun.
- No tokens were billed for these attempts (400 pre-generation validation errors, not completions).
- No Anthropic raw response files were cached (failed calls are never written to the cache), so a rerun after 2026-10-01 00:00 UTC will cleanly attempt all 20 Anthropic calls fresh with no wasted/duplicate spend.

## Pipeline validation checklist (per protocol)

| Check | Status |
|---|---|
| Prompt formatting (no CoT, no dataset's own prompt/prompt_think reused) | PASS — custom template, verified offline before any API call |
| Answer parsing (separate from inference) | PASS — `phase1a_parse.py`, unit-tested offline on edge cases (wrapped labels, ambiguous multi-match, garbage) before spending |
| Label mapping (mcq A-D vs roman_numeral I-IV) | PASS — verified on real Gemini outputs, all correctly mapped |
| API stability | Gemini: stable. Claude: blocked by account cap (not a stability issue) |
| Caching | PASS — verified via 2nd run showing 20/20 cache hits, 0 duplicate Gemini calls/spend |
| Restart behavior | PASS — a schema-migration bug was caught and fixed mid-run (see below) without losing or duplicating any successful call |
| Raw response preservation | PASS — full raw API response JSON preserved per call under `results/behavioral/raw/<provider>/<model>/` |

**One real bug found and fixed during this smoke test**: the raw-record schema changed mid-development (from a `temperature` field to a `sampling_config` field) after some Gemini responses were already cached under the old schema. The parse step initially crashed reading old-schema cache files. Fixed with a backward-compatible `.get()` fallback rather than deleting/re-fetching the already-paid-for cached responses. This is exactly the kind of failure the smoke test is meant to surface before it happens at 200-question scale.

## Estimated remaining API requests for full Phase 1A

- Full cohort: 200 originals × 2 conditions (mcq, roman_numeral) × 2 models = **800 total requests**.
- Already attempted in this smoke test: 40 (20 succeeded/Gemini, 20 blocked/Claude).
- Remaining after smoke test, assuming Claude unblocks: 190 originals × 2 × 2 = **760 requests**, plus the 20 Anthropic calls still owed from this smoke test = **780 requests remaining** to complete the full 200-question, 2-model cohort.

## STOP

Per protocol: not proceeding to the remaining 190 questions. Additionally blocked from even completing the Claude half of the *smoke test itself* until the account usage cap resets (2026-10-01 00:00 UTC per Anthropic's own error message).
