# Behavioral Experiment Plan — Phase 1, Task 11

Project: NAACL 2027 medical-LLM-reliability paper. This document is a
**plan only** — no LLM API calls, no training, no GPU code were run to
produce it. It builds directly on the completed Phase 0 forensics
(`results/audit/dataset_audit.md`, `data/processed/PAIRING_METHOD.md`,
`data/processed/PILOT_COHORT.md`, `results/audit/perturbation_descriptives.csv`,
`results/audit/dataset_manifest.csv`) and on external verification of the
ReMedQA paper/repo (Section 1). Nothing here should be read as "already
run" — Sections 3–5 are a specification for someone (or an agent, once
explicitly authorized) to execute later.

---

## 1. Official ReAcc / ReCon definitions — source and exact text

**Source found and verified:**

- Paper: *ReMedQA: Are We Done With Medical Multiple-Choice Benchmarks?*,
  EACL 2026, ACL Anthology `2026.eacl-long.124`
  (https://aclanthology.org/2026.eacl-long.124/, PDF at
  https://aclanthology.org/2026.eacl-long.124.pdf).
- Project page: https://disi-unibo-nlp.github.io/remedqa/
- Dataset card: https://huggingface.co/datasets/disi-unibo-nlp/ReMedQA

**Metric definitions, as stated by the paper's abstract and project page**
(quoted, not paraphrased from memory):

- **ReAcc**: "measures if a model gets the correct answer for all versions
  of a question" — i.e., **the proportion of ORIGINAL questions for which
  the model is correct on every one of its variants** (strict, all-or-
  nothing per original question — not a per-row accuracy average).
- **ReCon**: "measures if a model gives the same answer to all versions of
  a question" — i.e., **the proportion of original questions for which the
  model's answers are self-consistent across all variants, regardless of
  whether that consistent answer is correct.**

**Perturbation set, per the project page** (5 "closed-format" perturbations
plus the open-format variant, matching what Phase 0 already found
independently on-disk): No Labels (`no_symbols`), Roman Numerals
(`roman_numeral`), Fixed Position (`fixed_pos`), Select Incorrect
(`incorrect`), None Provided (`none_of_the_provided`), plus the open-ended
format (`open`).

**Fixed-position design detail, quoted directly from the project page**
(this is load-bearing for Section 3 below): *"We always place the gold
option in position D. We select D because it is the least frequent gold
position in the datasets, minimizing positional priors while stressing
robustness to order."* This independently confirms, from the paper's own
design description, the Phase 0 finding that `fixed_pos` is a **deterministic
placement at a fixed target position, not a random reshuffle**.

**What this means for the plan:** ReAcc and ReCon are real, officially
defined, paper-grounded metrics, and this plan adopts them (Section 4) as
first-class outputs, defined precisely against our own instance set.
Everything else this plan computes in addition (per-perturbation accuracy,
flip rates, stable-correct/stable-wrong rates) is **project-defined**, not
from the original paper, and is labeled as such below — it exists because
ReAcc/ReCon are coarse (all-or-nothing across variants) and a directional
breakdown (which specific perturbation, which direction of flip) is needed
for diagnosis, not just a summary reliability number.

**One open item, honestly flagged:** the search and fetches performed here
did not surface the original paper's exact table of which models were
evaluated, nor whether Claude and/or Gemini specifically were in that set,
nor whether the paper reports significance tests alongside ReAcc/ReCon.
**UNCERTAIN: this should be checked against the full paper PDF (Section 5,
"Experiments," and the model list table) before claiming any
Claude/Gemini-specific novelty** — see Section 6, adversarial question 10.

---

## 2. Scope: Claude + Gemini now; open-weight model deferred

This plan covers **behavioral (black-box, API-level) evaluation of Claude
and Gemini only.** A later phase is expected to add an open-weight model
(e.g. a Llama/Qwen/Mistral-class model run locally) specifically so that
**representation-level analysis** (hidden-state probing, attention
inspection, logit-lens style diagnostics on *why* a perturbation flips an
answer) becomes possible. That phase is **out of scope here** and is not
speced in any technical detail beyond this paragraph, because:

- It requires GPU inference and access to model internals, neither of which
  a hosted Claude/Gemini API exposes.
- It is a materially different kind of experiment (mechanistic, not
  behavioral) that should be designed only after the behavioral pilot in
  this document produces results worth explaining mechanistically —
  designing the probing methodology now would be premature and likely
  wrong once we know where Claude/Gemini actually flip.
- This document's guardrails (no GPU code, no training) explicitly exclude
  it.

---

## 3. Evaluation instances

### 3.1 Source table

Use `data/processed/pilot_200.parquet` (1,200 rows: 200 stratified
original questions — medqa 80, medmcqa 63, mmlu 57, seed=42 — × 6 non-`mcq`
perturbations each) for the first pass, plus one `mcq`-baseline call per
original question (pulled by `(source_dataset, source_id)` from
`data/processed/remedqa_pairs.parquet`'s baseline fields, or directly from
`data/raw/remedqa`'s `{source}_mcq` splits). This is the same 200-question
cohort already built and documented in `PILOT_COHORT.md`; this plan does
not resample.

### 3.2 Recommended primary perturbation scoping — and why

**This is a design recommendation for the user to confirm, not a
unilateral final decision**, and it explicitly **narrows** the boundary
dataset_audit.md itself drew in §4/§8. That prior document classified
`roman_numeral` and `fixed_pos` as "the only clean meaning-preserving
perturbations" and `none_of_the_provided` as not meaning-preserving at all.
This plan recommends a different split, for reasons that were not fully
available when that audit section was written:

| Perturbation | Recommended bucket | Rationale |
|---|---|---|
| `roman_numeral` | **Primary meaning-preserving set** | 0% question-text change, option text/order/content byte-identical to `mcq` (`perturbation_descriptives.csv`: `question_text_changed_rate=0`, `lexical_similarity_mean=1.0`). Only the label alphabet changes. |
| `none_of_the_provided` | **Primary meaning-preserving set** | 0% question-text change; three of four option texts and the answer-output channel (single letter) are unchanged; only the correct option's text is swapped for a fixed placeholder that remains labeled correct. **Divergence from dataset_audit.md's stricter framing, flagged explicitly**: the audit treated any change to the correct option's semantic content as disqualifying. This plan treats it as meaning-preserving of the *task format* (still select-one-letter-of-four, same question, same distractors, same answer channel) while acknowledging a real residual confound (see adversarial Q4, "none of the above" response bias) — the user may prefer the audit's stricter reading and exclude it too. |
| `no_symbols` | **Primary meaning-preserving set** | 0% question-text change (verified directly, not assumed — `PAIRING_METHOD.md` and the manifest both note this contradicts the plausible-but-wrong assumption that it rewords like `open`). Only the answer-output channel changes (full option text instead of a letter). |
| `open` | **Separate category — different task/answer format** | 95.1% of questions reworded (`question_text_changed_rate=0.951`), answer becomes free text. Cannot be string-matched against a letter; needs its own scoring path (Section 3.4) and must not be pooled into the primary meaning-preserving statistics. |
| `fixed_pos` | **Excluded from primary analysis by default — treated as a separate position-bias probe** | See 3.3 below. This is a genuine **reversal** of dataset_audit.md's §4/§8 classification of `fixed_pos` as one of the two "clean" meaning-preserving perturbations, driven by evidence dataset_audit.md did not have: the paper's own design statement (Section 1) that gold is placed at D in 100% of items, confirmed locally (see 3.3). |
| `incorrect` | **Excluded from primary scoring entirely** | Task-inverted (select-the-wrong-options, not select-the-right-one); `perturbed_answer_interpretation` is null by construction in the pairing table for this reason. May be run later as a distinct "can the model follow an inverted instruction" probe, never as a correctness-preserving comparison. |

So the **primary meaning-preserving perturbation set for the core
hypothesis is `{roman_numeral, none_of_the_provided, no_symbols}`**,
compared against the `mcq` baseline, with `open` analyzed separately
(different task shape) and `fixed_pos`/`incorrect` excluded from that
primary comparison and run as separate, explicitly-labeled probes.

### 3.3 `fixed_pos` — position-bias confound, flagged prominently

`perturbation_descriptives.csv` shows `fixed_pos`'s
`answer_position_change_rate = 0.7689` (76.89% of the 3,154 original↔
fixed_pos pairs have a different answer letter than the `mcq` baseline).
Combined with the paper's own stated design ("We always place the gold
option in position D"), this is fully explained and cross-validated: **100%
of `fixed_pos` items have their correct answer at option D**, and the
23.11% of items whose baseline `mcq` answer was *already* D are exactly the
ones with `answer_position_change_rate = 0` (i.e., $1 - 0.7689 = 0.2311$).
This is an internal consistency check this plan performed against numbers
already on disk, not a new query against the raw data.

**Why this disqualifies `fixed_pos` from the primary "meaning-preserving"
bucket**: a meaning-preserving perturbation should let us test "does the
model's answer change when nothing about the question's substance
changes." `fixed_pos` instead moves every single item to the position the
paper's authors *chose specifically because it is the dataset's rarest gold
position* — i.e., it is a deliberately adversarial, deterministic
manipulation of position, not a random meaning-preserving reshuffle. A
model that always answers "D" would score artificially high on the
*baseline* items that already have gold=D-ish frequency near expectation,
but a model with any latent position prior (for or against D) will show a
`fixed_pos` accuracy delta that is genuinely uninterpretable as "did the
model still understand the question" — it conflates content robustness
with position-prior robustness, and the design makes that conflation
maximal, not incidental.

**Recommended mitigation (this plan's choice, to be confirmed by the
user)**: exclude `fixed_pos` from the primary meaning-preserving accuracy/
consistency/ReAcc/ReCon analysis, and instead report it as a **separate,
explicitly-labeled position-bias probe** ("does the model disproportionately
answer/avoid D, and does forcing gold to the rarest natural position change
accuracy more than the other format perturbations do"). Re-randomizing
`fixed_pos` (building a new, actually-random position shuffle from the
`mcq` options) was considered and rejected for this pass: it would require
constructing new perturbed data not present in the on-disk ReMedQA dataset,
which is out of scope for a Phase 0/1 pilot that is supposed to use the
data as audited. If the user wants a true random-shuffle condition later,
that is a `data/processed`-level task, not a behavioral-eval task.

### 3.4 Constrained output format, per perturbation — concrete

All perturbations use a **project-authored minimal prompt template**, not
the dataset's own `prompt`/`prompt_think` fields verbatim — both of those
explicitly request step-by-step reasoning ("Solve them in a step-by-step
fashion...", "think step by step and then end with..."), which conflicts
with this plan's "no chain-of-thought requested" requirement (Section 4).
Reusing them would also make the `mcq` baseline's prompt differ in framing
from the custom perturbation prompts, confounding perturbation effects with
prompt-template effects (see adversarial Q1). Instead:

- **Template**: a single fixed instruction ("Answer the following question.
  Respond with only the requested field, no explanation.") + question +
  options (where applicable), identical in structure across `mcq`,
  `roman_numeral`, `none_of_the_provided`, `no_symbols`, and `open`, varying
  only in the option-label scheme / presence of options / output-field
  description each perturbation actually requires. The template's exact
  text and a hash of it are logged per call (Section 3.5) so it is
  versioned and reproducible.
- **`mcq`, `roman_numeral`, `none_of_the_provided`**: constrained
  single-choice output. Concretely: use each provider's structured-output
  mechanism — Claude: a forced tool call with one required parameter whose
  JSON Schema is `{"type": "string", "enum": [<item's own valid labels,
  e.g. ["A","B","C","D"] or ["I","II","III","IV"]>]}`; Gemini: the
  equivalent `responseSchema` with an `enum` constraint. This removes
  free-text parsing ambiguity entirely — the model cannot emit anything
  outside the valid label set for that item.
- **`no_symbols`**: no label exists to constrain against, so use a
  structured JSON output `{"answer_text": string}` (schema-constrained to a
  single string field, no enum possible since options aren't labeled).
  Score by normalized exact-string match (lowercase, strip punctuation,
  collapse whitespace — reuse the exact normalization already defined in
  `scripts/audit_datasets.py::normalize_question`, applied to answer text
  instead of question text) against the option text that is recorded as
  correct in that row's `perturbed_options`. Flag non-exact matches with
  high token overlap (e.g. token-set Jaccard ≥ 0.8, reusing the audit's own
  near-duplicate method) for manual spot-check rather than auto-scoring
  them either way, since paraphrase (e.g. "cross-linking of DNA" vs. "DNA
  cross-linking") is expected and a hard exact-match would undercount
  correct answers.
- **`open`**: same structured `{"answer_text": string}` output. Score
  against `perturbed_answer_interpretation` (already computed and stored in
  `remedqa_pairs.parquet`/`pilot_200.parquet` for exactly this purpose)
  using the same normalized-exact-match-first, Jaccard-flag-for-review
  approach as `no_symbols`. Because `open` questions are reworded (not just
  reformatted), semantic-equivalence judgment is harder here than for
  `no_symbols` — this plan explicitly does **not** propose an automated
  semantic-equivalence scorer (e.g., an LLM-judge pass) for this round,
  since that would itself be an LLM API call requiring its own validation;
  it flags a short human rubric pass over the `open` disagreement cases as
  a likely follow-up, not something to build now.
- **`incorrect`** (if run later, outside primary scoring): constrained
  multi-select — a tool/schema requiring an array of exactly 3 distinct
  labels from the item's valid label set — scored by exact set equality
  against the parsed `perturbed_answer_raw` list. Never scored as, or
  compared to, `original_answer_text`.
- **`fixed_pos`** (position-bias probe, if run): same constrained
  single-choice mechanism as `mcq`, scored against that row's own
  (D-placed) correct label — reported separately, not pooled.

### 3.5 Same instances across Claude and Gemini

Both models are run against the **identical `pilot_200` rows and the
identical prompt template** (Section 3.4), differing only in the
model-specific structured-output call mechanism (tool schema vs.
`responseSchema`) needed to get an equivalent constrained response — this
is a technical necessity, not a content difference, and the underlying
instruction text and options presented to the model are byte-identical
across providers. This directly satisfies "same evaluation instances used
across Claude and Gemini wherever technically possible."

### 3.6 Call budget for this pilot pass

Per model, per full pass: 200 (`mcq` baseline) + 200×3 (primary
meaning-preserving set) + 200 (`open`, separate scoring) + 200 (`fixed_pos`
probe) = **1,000 calls**, or **1,200** if `incorrect` is also run as a
separate probe. Across both Claude and Gemini: **2,000–2,400 calls total**
for one full pilot pass. This is stated here for planning purposes only —
no calls have been made.

### 3.7 Raw outputs to log, and where

**Path**: `results/behavioral/` (to be created only when the experiment is
actually executed — not created by this planning task, per instructions).
Recommended layout: one row per (model, source_dataset, source_id,
perturbation_type) in a single append-only table, e.g.
`results/behavioral/raw_responses.parquet`, with columns:

- `model_name` (exact API model identifier string, including version tag)
- `provider` (`anthropic` / `google`)
- `timestamp` (UTC, ISO 8601, call time)
- `prompt_template_version` / `prompt_template_hash` (sha256 of the exact
  rendered prompt string sent)
- `source_dataset`, `source_id`, `perturbation_type` (join keys back to
  `pilot_200.parquet`)
- `raw_response` (full unmodified API response text/JSON, including any
  structured-output wrapper)
- `parsed_answer` (the extracted label or answer text after schema
  parsing)
- `correctness` (bool or null for `incorrect`/ambiguous `open` cases
  pending manual review)
- `latency_ms`
- `api_metadata` (JSON blob: token usage, stop reason, request id, any
  provider-side safety/refusal flags, temperature/seed actually echoed
  back if the API confirms it)
- `run_id` (groups a full pilot pass together, for reproducibility across
  reruns)

---

## 4. Deterministic inference settings

- **Temperature**: 0 (or the provider's documented floor if 0 is not
  literally supported) for both Claude and Gemini.
- **Seed**: pass a fixed seed on any endpoint that accepts one (Gemini's
  API supports a `seed` parameter; if Claude's API does not expose one at
  the time this runs, log `seed=unsupported` explicitly rather than
  silently omitting the field). **Honest caveat, stated here rather than
  glossed over**: temperature=0 plus a fixed seed does **not** guarantee
  bit-identical outputs on hosted, batched inference services — both
  providers document that determinism is best-effort, not guaranteed, due
  to floating-point non-associativity under dynamic batching. This plan
  treats "deterministic settings" as variance-*minimizing*, not
  variance-*eliminating*, and does not claim exact reproducibility as a
  result.
- **No chain-of-thought requested**: enforced by the custom prompt template
  (Section 3.4), which explicitly asks for only the constrained answer
  field, and by disabling any extended-thinking mode the API exposes (e.g.
  Claude's extended thinking should be off for this pass).
- **Constrained output**: schema/tool-enforced per perturbation as detailed
  in Section 3.4 — this is the mechanism that makes "parse the model's
  answer" a solved problem rather than a regex-guessing exercise, except
  for the two free-text perturbations (`no_symbols`, `open`) where
  normalized string matching plus a manual-review flag band is used.

---

## 5. Computable metrics

All defined below are computed **per model**, first on the primary
meaning-preserving set `{roman_numeral, none_of_the_provided, no_symbols}`
vs. `mcq`, then separately for `open` vs. `mcq` and for the `fixed_pos`
probe vs. `mcq`.

**Paper-grounded (Section 1):**
- **ReAcc** = (number of original questions where the model is correct on
  `mcq` **and** correct on every included perturbation in the set being
  analyzed) / (number of original questions in that set).
- **ReCon** = (number of original questions where the model gives a
  consistent answer — same selected option/content — across `mcq` and
  every perturbation in the set, regardless of correctness) / (number of
  original questions). Note "consistent" must be defined per answer
  channel: for label-based perturbations, consistency means the same
  *option content* was selected (not the same *letter*, since labels
  change under `roman_numeral`); for `no_symbols`/`open`, consistency means
  the selected option content matches (via the same normalized-match
  criterion used for correctness).

**Project-defined (not from the original paper, labeled as such):**
- **Accuracy per perturbation type**: correct / n, computed independently
  for `mcq` and each perturbation.
- **Perturbation consistency (pairwise)**: for a given non-baseline
  perturbation, the fraction of original questions whose answer content
  under that perturbation matches the answer content under `mcq`
  (regardless of correctness) — this is ReCon's pairwise, per-perturbation
  decomposition rather than the all-variants-at-once version.
- **Correct→wrong flip rate**: among originals correct on `mcq`, fraction
  that become incorrect under a given perturbation.
- **Wrong→correct flip rate**: among originals incorrect on `mcq`, fraction
  that become correct under a given perturbation.
- **Stable-correct rate**: fraction of originals correct on both `mcq` and
  the perturbation.
- **Stable-wrong rate**: fraction of originals incorrect on both.

---

## 6. Statistical plan

- **Paired binary correctness comparisons → McNemar's test, not a paired
  t-test.** For a given (model, perturbation) pair, build the 2×2 table
  over the same 200 (or per-source-subset) original questions:

  |  | Perturbed correct | Perturbed incorrect |
  |---|---|---|
  | **Original (`mcq`) correct** | a (stable-correct) | b (correct→wrong) |
  | **Original (`mcq`) incorrect** | c (wrong→correct) | d (stable-wrong) |

  McNemar's test uses only the discordant cells $b$ and $c$ to test the
  null hypothesis that perturbation does not systematically change
  correctness (marginal homogeneity). Given the pilot's modest per-cell
  counts (n=200 originals, likely fewer than that in the discordant cells),
  use the **exact binomial form** of McNemar's test rather than the
  chi-square approximation, which is unreliable when $b+c$ is small.
- **Bootstrap confidence intervals for accuracy/consistency proportions —
  resample at the ORIGINAL-QUESTION level.** Concretely: resample, with
  replacement, whole `(source_dataset, source_id)` clusters (200 draws per
  bootstrap iteration, standard number of iterations e.g. 2,000–10,000),
  and for each resampled cluster carry along **all** of its associated rows
  (its `mcq` baseline plus every perturbation row included in the analysis
  being bootstrapped). Recompute the statistic on each resample. **Why
  cluster-level, not row-level**: rows within a cluster share the same
  underlying question (same clinical vignette, same difficulty, same
  knowledge item) and are not independent draws — a row-level bootstrap
  would treat, e.g., a `roman_numeral` row and its `mcq` sibling as two
  independent observations, understating true variance and producing
  falsely narrow confidence intervals for a design that is fundamentally
  paired/clustered. Resampling whole originals with their perturbation
  family attached respects the actual unit of random sampling used to
  build the pilot (`PILOT_COHORT.md`: 200 originals were the sampling
  unit, not 1,200 rows).
- **Explicitly deferred, not computed in this phase**: AUROC, AUPRC,
  Brier score, ECE, risk-coverage curves, and AURC. All of these require a
  per-instance confidence/probability signal, which this design does not
  yet elicit (no logprob or verbalized-confidence field is requested in
  Section 3.4's constrained-output schema). They belong to a later phase
  once a confidence-elicitation protocol exists — adding a confidence field
  now, without validating it, would produce numbers that look rigorous but
  aren't grounded in anything this plan has actually verified.

---

## 7. Adversarial section

**1. What alternative explanations exist for an accuracy drop under
perturbation, besides "the model doesn't really understand the
question"?** Output-format/parsing mismatch is the biggest one even with
schema-constrained outputs — a model could refuse to comply with the
schema, or a harness bug in mapping perturbation-specific label sets could
silently miscount. Prompt-template quality is a second: this plan
deliberately does *not* reuse the dataset's own `prompt`/`prompt_think`
fields (Section 3.4) specifically to avoid conflating a home-grown minimal
template's effect with the perturbation's effect — but that also means our
`mcq` baseline accuracy is not directly comparable to any number the
original paper reports for the *same* models using the *dataset's own*
prompts, since prompt engineering itself affects accuracy.

**2. What dataset artifacts could confound the results?** MMLU's 136/895
(~15%) within-split duplicate-stem rate (dataset_audit.md §6) means the
57-item mmlu stratum of the pilot could contain near-identical stems whose
errors are correlated rather than independent — this weakens the effective
sample size for the mmlu subgroup specifically and should be checked (are
any of the 57 sampled mmlu originals duplicates of each other?) before
mmlu-subgroup claims are made.

**3. Answer-position bias — cite the `fixed_pos` finding specifically.**
Already covered in depth in Section 3.3: `fixed_pos` places gold at
position D for 100% of items, by the paper's own explicit design choice,
specifically because D is the *rarest* natural gold position in the
underlying data (independently cross-validated here: 23.11% of pilot-scale
originals already have gold=D in `mcq`, consistent with "rarest of four,"
which would be ~25% under a uniform distribution and is observed to be
below it). This is why `fixed_pos` is excluded from the primary
meaning-preserving bucket rather than merely flagged: it isn't a neutral
random perturbation, it's an adversarially chosen fixed target.

**4. Lexical-change confounds?** `none_of_the_provided` changes the
correct option's text to a fixed, generic placeholder string ("None of the
provided options"), with a small mean options-length shift
(`options_length_diff_mean_words=0.296` per `perturbation_descriptives.csv`).
Beyond lexical length, this specific placeholder is a well-known trigger
for meta-option response biases documented in MCQ literature (models may
be systematically more or less willing to select "none of the above"-style
options independent of whether it's actually correct) — an accuracy/
consistency shift under this perturbation could reflect that general bias
rather than "did the model still understand the question," which is a real
weakness of including it in the primary set (see the explicit divergence
noted in Section 3.2) and a reason the user might reasonably override this
plan's recommendation and drop it back to a task-format-change bucket.

**5. Formatting confounds?** `no_symbols` and `open` both require
free-text scoring via normalized string match plus a fuzzy-match review
band (Section 3.4). Any "incorrect" score under these two could be a
scoring artifact (a correct paraphrase the normalizer didn't catch) rather
than a genuine model failure — this is why Section 3.4 mandates a
manual-review flag band rather than trusting the automated score outright
for these two perturbations specifically.

**6. Is this model-specific, and does that limit interpretation?** Yes,
unavoidably. Claude and Gemini differ in training data, cutoff, RLHF
tuning, and structured-output implementation; MedQA/MedMCQA/MMLU are public
benchmarks, so some amount of memorization (rather than reasoning) likely
contributes to both models' `mcq` baseline accuracy, and that contribution
cannot be disentangled from genuine understanding without a separate
memorization probe, which this plan does not include. Results here
describe *these two specific hosted models' behavior in September 2026*,
not language models in general.

**7. Is the pairing actually correct?** Yes — this is the one pillar of
the whole study that is genuinely solid, not just asserted. `PAIRING_METHOD.md`
and dataset_audit.md §5 both document that id-based pairing was verified
(not assumed): zero duplicate ids within any split, union of ids equals
intersection of ids across all 7 perturbations per source (medqa 1259/1259,
medmcqa 1000/1000, mmlu 895/895), and `build_pairs.py` re-verifies this
programmatically at run time with a hard failure if it doesn't hold. The
200-question pilot inherits this guarantee directly since it's a filtered
subset of the same verified table.

**8. Is the bootstrap plan actually robust at this sample size?** Partially.
Cluster-level bootstrap (Section 6) is the statistically correct choice
given the paired/clustered structure, but with only 200 clusters overall —
and as few as 57 within the mmlu stratum — confidence intervals,
especially for any per-source breakdown, will be wide, and rare-event
proportions (e.g., a low flip rate on a robust perturbation) may have
bootstrap CIs that include zero or are asymmetric. Per-source subgroup
findings from this pilot should be treated as exploratory, not
confirmatory, until scaled to the full 3,154-original `remedqa_pairs.parquet`
table.

**9. Does ReMedQA's own paper already establish this?** Largely, yes, and
this should be said plainly rather than downplayed. The paper's central,
already-published finding (Section 1) is precisely that high MCQA accuracy
masks low reliability and that models — including medical-specialized ones
— remain sensitive to format/perturbation changes, measured via ReAcc/
ReCon across (per the abstract) multiple models. A bare replication of
"Claude/Gemini accuracy drops under perturbation" would not, by itself, be
new. See item 10 for where genuine novelty would have to come from.

**10. What NEW knowledge would this specific design provide, beyond
"accuracy drops under perturbation"?** Three candidate sources of novelty,
none confirmed yet: (a) **Model coverage** — whether Claude and Gemini
specifically were in the original paper's evaluated model set is
**UNCERTAIN** from the sources checked here (the abstract/project page did
not enumerate models); if they weren't, this pilot adds direct coverage the
paper doesn't have. This must be checked against the full paper before
claiming novelty. (b) **Granularity** — this plan's flip-rate/stable-
correct/stable-wrong breakdown (Section 5) and McNemar+cluster-bootstrap
inferential layer (Section 6) are more granular and more statistically
rigorous than a single reported ReAcc/ReCon point estimate, *if* the
original paper does not already report an equivalent breakdown — also
unverified here. (c) **Positioning for the deferred open-weight phase**
(Section 2) — a black-box behavioral study, by construction, cannot explain
*why* an answer flips; this pilot's real strategic value is as the
precondition for the later representation-level analysis, which the
original ReMedQA paper does not and cannot provide. Absent (a) or (b) being
confirmed favorable, (c) is the most defensible claim to novelty this
design currently supports.

---

## 8. Final decision-tree classification: **YELLOW**

**Not GREEN**, because real, material issues were found that require
explicit exclusions/re-scoping before any experiment can run cleanly:

1. **`fixed_pos` is not a random, meaning-preserving shuffle** — it
   deterministically places gold at the dataset's rarest natural position
   (paper-confirmed design choice, cross-validated locally), which is a
   serious confound for the study's core "meaning-preserving perturbation"
   hypothesis and required this plan to reverse dataset_audit.md's original
   classification of it as "clean." It must be excluded from primary
   scoring and re-labeled as a separate probe (Section 3.2–3.3).
2. **Local MedQA's `test` split is contaminated relative to ReMedQA** —
   98.9%/100% overlap with `medqa_mcq` (dataset_audit.md §7) means it
   cannot be used as an independent evaluation or replication set for
   anything touching ReMedQA content; only `train`/`validation` (0%
   overlap) remain usable if a fresh MedQA-sourced cohort is ever needed.
3. **`incorrect` and `open` require materially different handling, not
   just different prompts** — `incorrect` inverts the task and its answer
   field must never be scored as a correctness target; `open` changes both
   question phrasing (95.1% reworded) and answer format (free text),
   requiring a different, less-reliable scoring path (normalized match +
   manual review band) than the rest of the perturbations.

**Not RED**, because the one thing the entire study structurally depends
on — reliable id-based pairing of an original question to all of its
perturbed variants — **is solid**, verified twice (forensic audit +
independent programmatic re-verification with a hard-fail check in
`build_pairs.py`), with zero gaps or mispairings found across all 3,154
originals. CareQA is additionally confirmed fully independent (0% overlap,
0 near-duplicates against both ReMedQA and MedQA) and stands as a genuine
future held-out validation set. None of the issues found are fatal to the
study; they are exactly the kind of scoping decisions a plan is supposed to
make explicit before data collection, and this document makes them
explicit rather than letting them surface as an uninterpretable pooled
result later.

**Net**: proceed to execution, but only under the exclusions and
scoping choices specified in Sections 3–6 above — not as a naive "run all
7 perturbations, average everything" pass.

---

*Produced as Phase 0/1 planning (Task 11). No LLM API calls, model
training, or GPU code were executed to produce this document. External
verification used WebSearch/WebFetch against the ACL Anthology page/PDF,
the ReMedQA project page, and the HuggingFace dataset card, as cited in
Section 1.*
