# Phase 2A — Single-Model End-to-End Framework Validation

**Scope:** framework validation only. gemini-3.8-flash, N = 50 underlying questions × 7 ReMedQA
representations = 350 evaluations. Not the final benchmark; n=50 supports no statistical or
model-superiority claim.

Config hash `fadf6ef4fda8865d…` · prompt `phase2a_v1` · parsers `phase2a_parsers_v1` ·
evaluators `phase2a_evaluators_v1` · thinking_budget 1024, max_output_tokens 1536 (shared ceiling),
temperature 0, JSON-schema constrained output. Full provenance: `run_manifest.json`.

## 1. Semantic audit (pre-run, all 350 rows)

0 contradictions against the registry, checked both on example contents and on ReMedQA's own
dataset prompts (displayed options, label style, answer format, task inversion). All questions have
4 options; no question has duplicate option texts. Details: `semantic_audit_summary.md`.

| condition | verified operational meaning |
|---|---|
| mcq | baseline, letter labels |
| roman_numeral | labels → I–IV only; question, option texts, order, gold position/content unchanged (50/50) |
| none_of_the_provided | gold text → "None of the provided options"; gold position unchanged (50/50); original correct content removed (50/50) |
| fixed_pos | same option set, reordered, gold always at D; gold position moved in 42/50 (8 were already at D), content unchanged |
| incorrect | same options; gold = the 3 non-correct letters; dataset prompt inverts the task |
| open | options NOT shown; question reworded in 47/50; free-text answer, gold = mcq gold text |
| no_symbols | same options in same order as unlabeled bullets; answer = option text |

## 2. Framework architecture

```
src/medperturb/
  schemas/response.py        Example, OutputSpec, ModelResponse, ParseResult, EvalResult
  perturbations/registry.py  PerturbationSpec per condition (category, display, output kind,
                             evaluator, transition type, which paired metrics are justified)
  perturbations/audit.py     pre-run semantic audit gate
  prompts/templates.py       answer-only templates (Phase 1 template preserved byte-for-byte)
  adapters/base.py           ModelAdapter contract + provider-neutral JSON schema
  adapters/gemini.py         the only Gemini-specific code (SDK, tokens, finish reason, pricing)
  parsing/canonical.py       parser registry: label / label_set / free_text
  evaluation/semantic.py     evaluators: position identity AND semantic (option-text) identity
  evaluation/cache.py        config-hash + prompt-hash cache, atomic writes, secret guard
  evaluation/runner.py       resumable runner, duplicate-call guard, retry, cost ceiling
  evaluation/metrics.py      responses / pairs / per-perturbation metrics / transitions / failures
  evaluation/qc.py           structural assertions + independent plain-Python recomputation
scripts/phase2a_cohort.py, phase2a_run.py (--stage 1|2|3), phase2a_figures.py
```

Adding a model means writing one adapter. No evaluation code is Gemini-specific. The label
parser was regression-tested against all 80 cached Phase 1 responses: 0 mismatches.

## 3–8. Execution

| | |
|---|---|
| Expected evaluations | 350 |
| New API calls | 350 total, each (question, condition) called exactly once: Stage 1 = 7, Stage 2 = 28, Stage 3 = 315. (The stage1/ manifest records 0 new calls because its analysis was re-run from cache after a pairing-code fix.) |
| Cache hits in Stage 3 | 35 (all Stage 1/2 results, identical config hash) |
| Provider failures | 0 final (1 transient 503, recovered on the single allowed retry) |
| Parse failures | 0 / 350 |
| Truncations (MAX_TOKENS) | 0 / 350 |
| Thinking-cap saturation | 1 / 350 (none_of_the_provided, answered correctly, finish STOP) |
| QC / independent recompute | all passed, 0 discrepancies |

## 9. Metrics per perturbation (never pooled)

| condition | N | parsed | evaluable | accuracy | ReAcc vs mcq | ReCon vs mcq |
|---|---|---|---|---|---|---|
| mcq | 50 | 50 | 50 | 98% (49) | — | — |
| roman_numeral | 50 | 50 | 50 | 98% (49) | 0.98 | 1.00 |
| fixed_pos | 50 | 50 | 50 | 96% (48) | 0.96 | 0.98 |
| no_symbols | 50 | 50 | 50 | 98% (49) | 0.98 | 1.00 |
| none_of_the_provided | 50 | 50 | 50 | 88% (44) | not defined | not defined |
| incorrect | 50 | 50 | 50 | 98% (49) | not defined | not defined |
| open | 50 | 50 | **15** | undefined; bounds 30%–100% | not defined | not defined |

Joint across the meaning-preserving set {mcq, roman_numeral, fixed_pos, no_symbols}:
ReAcc_all = 0.96, ReCon_all = 0.98 (50/50 questions evaluable in all four).

ReCon is deliberately not computed for none_of_the_provided (the correct content no longer
exists), incorrect (a different task), or open (a provisional evaluator).

## 10. Position vs semantic answer change (vs mcq)

| condition | position change | semantic change | reading |
|---|---|---|---|
| roman_numeral | 0% | 0% | label swap only |
| fixed_pos | **82%** | **2%** | position changes are required by the reorder; only 1 real semantic change |
| no_symbols | 0% | 0% | |
| incorrect (implied answer) | 0% | 0% | inverted task answered consistently with mcq |
| none_of_the_provided | 10% | **98%** | semantic change is *required* (the gold content was replaced) |

This is the empirical case for tracking both identities: an index-based metric would report 82%
"instability" on fixed_pos that is not instability, and 10% on none_of_the_provided where the
decision actually changed in 98% of cases.

## 11. Transitions / adaptation

- roman_numeral, no_symbols, incorrect: 49 C→C, 1 W→W each (the same question: see §14).
- fixed_pos: 48 C→C, **1 C→W**, 1 W→W.
- none_of_the_provided: 44 successful adaptations (chose "None of the provided options");
  5 failures that chose a substantive distractor; 1 failure that kept its (already wrong) mcq distractor.
- incorrect: 49 successful (exact complement set), 1 failure that included the gold option, which
  is the same question the model misses on mcq, so it is consistent rather than an inversion failure.

Failure taxonomy (per response): 35 evaluation_ambiguity (all open), 6 required_adaptation_failure,
5 knowledge_error, 1 representation_instability. No provider, truncation, parse, invalid-option or
instruction-following failures.

## 12. Token usage (mean thinking + visible per response)

mcq 104+9 · roman 106+9 · fixed_pos 116+9 · no_symbols 75+15 · **none_of_the_provided 285+9**
(~2.7× mcq; max 1024) · incorrect 124+18 · open 63+16. Some responses report no thinking field;
the token arithmetic (total = prompt + candidates) verifies these are true zeros, recorded as such.

## 13. Cost

Phase 2A new spend: **$0.2208** (Stage 1 $0.0047, Stage 2 $0.0132, Stage 3 $0.2028), with thinking
tokens billed as output.

**Correction to Phase 1:** Phase 1 cost estimates omitted billed thinking tokens. Recomputed from the
cached raw responses, Phase 1's Gemini 3.x spend was **$0.0545**, not the reported $0.0097 (5.6×
underestimate). Cumulative project spend is therefore ≈ $0.28.

## 14. Examples worth manual review

1. **open — evaluator abstains on answers that look semantically correct.** "Urinary catheterization"
   vs gold "Insert a 'straight cath' into the patient's bladder"; "Intravenous normal saline
   hydration" vs "Intravenous fluids"; "Cerebral venous sinus thrombosis" vs "Sagittal sinus
   thrombosis". Also "Influenza A virus" vs "Rotavirus": with options hidden, influenza A also
   undergoes reassortment, so the reworded question may admit it. The 15 evaluable cases are
   exact-match survivors, so their 100% accuracy is survivorship-biased; report only the 30–100% bounds.
2. **mmlu::anatomy-023** — wrong identically under all 6 option-bearing representations ("mesoderm
   formation" vs gold "ectomesenchyme formation"). Consistent knowledge error, or a questionable gold
   label; worth checking.
3. **medmcqa::3c6acd9a** — the only representation instability: correct on mcq, "Can be used safely"
   vs gold "Low safety" under fixed_pos. It also fails none_of_the_provided.
4. **none_of_the_provided failures** (clinical_knowledge-062, 3c6acd9a, 4282ea27, medqa::0789,
   medqa::1144): the model picks a plausible distractor over the correct "none". Thinking use for
   these ranged 0–725 tokens, well below the cap, so they are not budget artifacts.

## 15. Completion

All 50 × 7 = 350 evaluations completed, parsed, and passed structural and independent QC checks.

## 16. Ready to freeze?

**Core framework: yes.** Registry, audit gate, prompts, adapter contract, label and label-set parsers,
semantic identity, cache and resume, metrics, and QC behaved correctly at N = 50, with no
parse or truncation failures and exact QC agreement.

**Not frozen: the open evaluator.** A deterministic string matcher cannot score free-text medical
answers. It is honest (it abstains rather than mis-scoring), but it leaves open unmeasured. This
needs a design decision: human annotation, a separate judge model, or a calibrated semantic-similarity
method.

## 17. Safe to scale to N = 200 with more adapters?

**Mechanically yes for Gemini:** about 1,400 calls at ~$0.0006 each is roughly $0.9. Before adding
models:
- Build and smoke-test a Claude adapter. Its API has no `temperature`; record `effort` instead. Its
  account was still capped at the last check.
- Settle the open-evaluation method first, or explicitly exclude open from the 200-question metrics.
- Keep monitoring thinking-cap saturation for none_of_the_provided (1/50 at N = 50).
