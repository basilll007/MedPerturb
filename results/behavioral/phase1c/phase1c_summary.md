# Phase 1C — Generation-Budget Control — Summary

Generated: 2026-09-29T23:27:24.309112+00:00

## 1. Reconciliation of the 13 previous parse failures

2 mcq + 1 roman_numeral (both Phase 1A, reused into Phase 1B) + 10 none_of_the_provided (Phase 1B) = 13/13 reconciled exactly. All 13 are Gemini-only (Claude never passed pre-flight in either phase, so it contributes zero rows either way). Full table: `parse_failure_audit.csv`.

## 2. Phase 1C inference configuration

`thinking_budget=1024`, `max_output_tokens=1536` (a single SHARED budget covering thinking + visible tokens together, confirmed via Gemini's own docs, not assumed). Phase 1B had effectively requested `thinking_budget=0` (not fully honored by these models - thoughts_token_count was observed >0 regardless) with `max_output_tokens=200`. Prompt template, models, and question set unchanged.

## 3. New API calls

0 new API calls made (20 cache-hit, 0 failed even after retry). Planned maximum was 20 (10 questions x 2 models); retries used: 0.

## 4. Phase 1B vs Phase 1C parse-success rate

- gemini-3.7-flash: 50% -> 100%
- gemini-3.8-flash: 50% -> 100%

## 5. Previously-failed responses recovered

10 / 10 recovered.

## 6. Remaining failures

0 still unparseable under the larger budget. See `phase1c_failure_recovery.csv` for exactly which.

## 7. Finish-reason distribution (Phase 1C)

{'FinishReason.STOP': 20}

## 8. Token usage before vs after

Phase 1B none_of_the_provided avg thinking tokens: ~157 (of 200 budget). Phase 1C avg thinking tokens: 250.8 (of 1024 budget); avg output tokens: 9.2 (of 1536 shared ceiling).

## 9. Canonical answer changes

All 10 questions x 2 models were rerun fresh under the new budget (not just the previously-failed ones) - full new answer set in `phase1c_responses.csv`. Among the 10 questions that parsed successfully in BOTH Phase 1B and Phase 1C, the answer changed in 0 case(s) - see `phase1c_failure_recovery.csv`.

## 10. Accuracy among parseable responses

           model  accuracy_among_parsed
gemini-3.7-flash                    0.8
gemini-3.8-flash                    0.9


### Important correction to Phase 1B's reported accuracy

Phase 1B reported 100% accuracy on none_of_the_provided, but that was computed on only the 5/10 responses per model that happened to complete before hitting the token budget - a **survivorship-biased subsample**, not the true rate. With all 10/10 now parseable under Phase 1C's budget, true accuracy is **17/20 (85%)**, not 100%. Notably, question(s) ['medmcqa::18c1a5f9-d998-414e-bb9c-991191c10710'] were answered incorrectly by **both** Gemini models with the **same wrong answer**, which is a substantive finding (the model prefers a plausible-sounding distractor over the correct-but-generically-labeled option), not a parsing/budget artifact - flagged for subsequent behavioral analysis per the Phase 1C interpretation guidance, since it persists despite an adequate budget.

## 11. Incremental cost

**$0.0025 USD**

## 12. Are Phase 1B failures inference-budget artifacts?

YES, evidence supports this: 10/10 previously-unparseable none_of_the_provided responses became parseable purely by raising the shared thinking+output budget, with the prompt and model held fixed. Reclassifying these as budget-associated parse failures / inference-configuration artifacts rather than a genuine behavioral-instability signal, pending confirmation at larger scale.

## 13. Is Gemini technically ready for scaling?

Yes, for MEASUREMENT purposes on none_of_the_provided specifically: with the Phase 1C budget config adopted as the new default for this condition, truncation is no longer the dominant failure mode (0/20 vs 10/20 before). However, the fuller sample also surfaced a real accuracy signal worth carrying forward (see the accuracy-correction note above) - readiness here means 'the pipeline measures correctly,' not 'the model performs well.' Claude remains entirely untested and unavailable.