# Phase 1B — Validation + Perturbation Sensitivity Check — Summary

Generated: 2026-09-29T23:11:52.475853+00:00

1. Claude availability: UNAVAILABLE (Anthropic account usage cap) - not retried, not substituted
2. New API calls by model: {}
3. Reused Phase 1A responses: 60
4. Parsing failures: 13 / 60

## 5. Semantic audit findings (none_of_the_provided)

- Gold letter position changed in 0/10 questions (adaptation genuinely required only for these).
- Gold option TEXT replaced with generic 'None of the provided options' boilerplate in 10/10 questions.
- Structural anomalies: 0/10.
- Full detail: `results/behavioral/phase1b/none_semantic_audit.csv`.

## 6. Metrics per model and condition (roman_numeral and none_of_the_provided reported SEPARATELY)

| model            |   mcq_accuracy |   roman_accuracy |   roman_delta_from_mcq |   roman_flip_rate |   roman_ReAcc |   roman_ReCon |   roman_n_valid_pairs |   none_accuracy |   none_delta_from_mcq |   none_answer_change_rate |   none_transition_CC |   none_transition_CW |   none_transition_WC |   none_transition_WW |   none_n_valid_pairs |
|:-----------------|---------------:|-----------------:|-----------------------:|------------------:|--------------:|--------------:|----------------------:|----------------:|----------------------:|--------------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| gemini-3.7-flash |              1 |                1 |                      0 |                 0 |             1 |             1 |                     9 |               1 |                     0 |                         0 |                    5 |                    0 |                    0 |                    0 |                    5 |
| gemini-3.8-flash |              1 |                1 |                      0 |                 0 |             1 |             1 |                     8 |               1 |                     0 |                         0 |                    5 |                    0 |                    0 |                    0 |                    5 |

## 7. Every behavioral transition

Full paired table: `results/behavioral/phase1b/phase1b_pairs.csv`. Roman-numeral flips:
None observed.

none_of_the_provided answer changes:
None observed.

## 8. Required-adaptation success/failure cases

Adaptation was required for 0 (model, question) pairs.
None — no question in this sample required a different answer under none_of_the_provided.

Full category breakdown (including not-required questions): `results/behavioral/phase1b/phase1b_adaptation.csv`.

## 9. Pipeline anomalies

- Claude unavailable (account usage cap) - Claude-side conditions not evaluated this run.
- **none_of_the_provided triggers a much higher truncation-driven parse-failure rate** on the Gemini 3.7/3.8-flash thinking-floor models: 50% vs 10% on other conditions. Root-caused via the cached raw responses (zero extra spend): mean `thoughts_token_count` on none_of_the_provided is 157 vs 100 on mcq/roman_numeral, out of the 200-token budget - responses are getting cut off mid-preamble (e.g. raw text `'Here is the JSON requested'`, finish_reason=MAX_TOKENS) before the JSON answer is ever emitted. This is a genuine behavioral-instability signal from Objective B, not pipeline noise: the model appears to reason substantially more about a 'none of the provided options is correct' framing. Recommend raising max_output_tokens specifically for this condition (or globally) before any scaled none_of_the_provided run.
- 13 unparseable responses recorded, never guessed at.

## 10. Incremental Phase 1B cost

**$0.0023 USD** (new calls only, reused Phase 1A responses cost $0).

## 11. Cumulative Phase 1A + 1B cost

**$0.0072 USD** (Phase 1A: $0.0049, Phase 1B incremental: $0.0023).

## 12. Readiness for the full pilot

NOT fully ready for none_of_the_provided specifically: 50% of Gemini responses on this condition were truncated before completing (see Pipeline anomalies above). mcq and roman_numeral are validated end-to-end on Gemini (low parse-failure rate, caching/reuse confirmed, independent metric recomputation matches exactly). Fix: raise the output token budget for none_of_the_provided before scaling it. Claude remains entirely unvalidated pending account access - not ready to scale Claude on any condition until it can be smoke-tested.