# Phase 2A — Pre-run semantic audit

Rows audited: 350 (50 questions x 7 representations). Contradictions: **0**.

Each row is checked against the registry's claimed taxonomy AND ReMedQA's own dataset prompt (displayed options, label style, answer format, task inversion).

| condition | category | q text changed | gold content changed | gold position changed | observed display | observed format | expectation met |
|---|---|---|---|---|---|---|---|
| mcq | baseline | 0/50 | 0/50 | 0/50 | labeled | letter_label | 50/50 |
| roman_numeral | presentation / answer-label transformation | 0/50 | 0/50 | 0/50 | labeled | roman_label | 50/50 |
| none_of_the_provided | answer-set / decision-boundary transformation | 0/50 | 50/50 | 0/50 | labeled | letter_label | 50/50 |
| fixed_pos | answer-position diagnostic | 0/50 | 0/50 | 42/50 | labeled | letter_label | 50/50 |
| incorrect | task/instruction inversion | 0/50 | 0/50 | 0/50 | labeled | label_set | 50/50 |
| open | response-space / MCQ-to-generation transformation | 47/50 | 0/50 | n/a (no options shown) | none | text | 50/50 |
| no_symbols | output-interface / answer-format transformation | 0/50 | 0/50 | 0/50 | bullets | text | 50/50 |

## Operational meaning, as verified

- **mcq** — baseline. Original multiple-choice representation, letter labels.
- **roman_numeral** — presentation / answer-label transformation. Option labels A-D become I-IV; question, option texts and order unchanged.
- **none_of_the_provided** — answer-set / decision-boundary transformation. Gold option's text replaced by 'None of the provided options'; position unchanged. Same index != same semantic answer: the original correct content no longer exists.
- **fixed_pos** — answer-position diagnostic. Options reordered so the gold answer sits in the last position (D). Deterministic gold-at-D placement; semantic identity compared by option text, not index.
- **incorrect** — task/instruction inversion. Model must name ALL incorrect options instead of the correct one. Correct iff selected set == all non-gold options; implied answer = the unselected option.
- **open** — response-space / MCQ-to-generation transformation. Reworded question with no options shown; free-text answer. Deterministic string evaluator is provisional; unmatched answers are evaluation-ambiguous, not wrong.
- **no_symbols** — output-interface / answer-format transformation. Options shown as unlabeled bullets; answer must be the option text.

Option counts observed: {4: 350}. Questions with duplicate option texts (would break text-based semantic identity): 0.
