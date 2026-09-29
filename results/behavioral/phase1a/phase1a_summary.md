# Phase 1A Smoke Test v2 — Summary

Generated: 2026-09-29T19:17:20.124900+00:00

Models attempted: ['claude-haiku-4-5-20251001', 'gemini-3.8-flash', 'gemini-3.7-flash']
Models that passed pre-flight and ran: ['gemini-3.8-flash', 'gemini-3.7-flash']
Models skipped after failed pre-flight: ['claude-haiku-4-5-20251001']

Total API calls made (non-cached): see logs/phase1a_smoke_v2.jsonl (event=call_success)
Total response rows collected: 40
Total parsing failures: 3

## Metrics per model

| model            |   n_calls |   n_parse_failures |   parse_failure_rate |   baseline_accuracy_mcq |   perturbed_accuracy_roman_numeral |   accuracy_delta |   answer_flip_rate |   ReAcc |   ReCon |   n_pairs_both_parsed |   estimated_cost_usd |   cost_calls_with_known_tokens |
|:-----------------|----------:|-------------------:|---------------------:|------------------------:|-----------------------------------:|-----------------:|-------------------:|--------:|--------:|----------------------:|---------------------:|-------------------------------:|
| gemini-3.7-flash |        20 |                  1 |                 0.05 |                       1 |                                  1 |                0 |                  0 |       1 |       1 |                     9 |            0.0023925 |                             20 |
| gemini-3.8-flash |        20 |                  2 |                 0.1  |                       1 |                                  1 |                0 |                  0 |       1 |       1 |                     8 |            0.00249   |                             20 |

## Every answer-flip case

No answer flips observed in this smoke sample.

## Approximate API cost

Total estimated: **$0.0049 USD** (sum of per-call token-based estimates using pricing in `scripts/phase1a_smoke.py::PRICING`, sourced 2026-09-29; calls with missing token counts are excluded from this sum, not assumed zero-cost).

## Suspicious pipeline behavior noted

- Pre-flight failed for: ['claude-haiku-4-5-20251001'] (see logs for exact error).
- 3 unparseable/ambiguous responses were recorded, not silently treated as wrong.

## Recommendation

NOT fully ready to scale as-is: 2/3 models passed pre-flight. Resolve the skipped model(s) before committing budget to the full 200-question run.