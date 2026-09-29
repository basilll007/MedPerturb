# CareQA (English) Forensic Audit — Findings

Phase 0 audit only. No API calls, no training. All numbers below come
directly from the on-disk dataset at `data/raw/careqa_en`.

## Load

- Loaded from: `F:\NAACL 2027\data\raw\careqa_en`
- Splits present: ['test']

## Schema

- n_examples (test split): **5621**
- Column names and dtypes (from `datasets.Features`):

| column | dtype/type |
|---|---|
| `exam_id` | `Value('int64')` |
| `question` | `Value('string')` |
| `op1` | `Value('string')` |
| `op2` | `Value('string')` |
| `op3` | `Value('string')` |
| `op4` | `Value('string')` |
| `cop` | `Value('int64')` |
| `year` | `Value('int64')` |
| `category` | `Value('string')` |
| `unique_id` | `Value('string')` |

Full `Features` repr (raw, unmodified):

```
{'exam_id': Value('int64'), 'question': Value('string'), 'op1': Value('string'), 'op2': Value('string'), 'op3': Value('string'), 'op4': Value('string'), 'cop': Value('int64'), 'year': Value('int64'), 'category': Value('string'), 'unique_id': Value('string')}
```

## 5 Representative Example Records (raw, unmodified)

### Example (row index 0)

```json
{
  "exam_id": 26,
  "question": "In relation to iron metabolism and its control mediated by hepcidin, it is true that:",
  "op1": "The drop in partial oxygen pressure promotes the activation of the hypoxia-inducible factor (HIF), which increases the expression of hepcidin.",
  "op2": "The increase in serum iron or inflammation stimulates the synthesis of hepcidin in the liver, which negatively regulates the function of ferroportin.",
  "op3": "Hepcidin reduces intestinal iron absorption through the inactivation of the divalent metal transporter 1 (DMT1).",
  "op4": "In hereditary hemochromatosis type 1, mutations in the human hemochromatosis protein (HFE) cause an increase in the production of hepcidin.",
  "cop": 2,
  "year": 2024,
  "category": "Medicine",
  "unique_id": "0e2d7263-0f92-429d-8345-612348f07960"
}
```
### Example (row index 1405)

```json
{
  "exam_id": 185,
  "question": "With regard to the frequency of advice on tobacco use in the general adult population, it is true that:",
  "op1": "It is not necessary to re-question individuals over 25 years old, who it is noted in the medical history, have never smoked.",
  "op2": "It should be carried out every 7-10 months in smokers, coinciding with the assessment of the rest of the factors.",
  "op3": "This should be carried out in all cases whenever the patient comes for a consultation for any other reason.",
  "op4": "It should be performed at least every 3 years in smokers, coinciding with the assessment of the rest of the factors.",
  "cop": 1,
  "year": 2022,
  "category": "Nursing",
  "unique_id": "ee6842bd-878b-4378-94ef-abe59cd32d81"
}
```
### Example (row index 2810)

```json
{
  "exam_id": 62,
  "question": "Indicate which of the following structures can never be observed in the umbilical cord:",
  "op1": "Mesenchymal cells.",
  "op2": "Artery.",
  "op3": "Trophoblast.",
  "op4": "Vein.",
  "cop": 3,
  "year": 2024,
  "category": "Biology",
  "unique_id": "dc6d6673-4bc1-4d9f-9994-e29d60accbb7"
}
```
### Example (row index 4215)

```json
{
  "exam_id": 95,
  "question": "The delusional atmosphere is a disturbance characterized as:",
  "op1": "A delusional secondary idea about the meaning of the world.",
  "op2": "A primary delusional idea in which the person has the experience that the world has changed.",
  "op3": "A secondary delusional idea in which the person varies their interpretations of perceptions.",
  "op4": "A primary delusional idea associated with the delusional reconstruction of memories.",
  "cop": 2,
  "year": 2022,
  "category": "Psychology",
  "unique_id": "2675b16d-1277-4a82-af5b-d73aa4dae727"
}
```
### Example (row index 5620)

```json
{
  "exam_id": 185,
  "question": "The mass balance of a saturated aqueous solution of the salt Ag3PO4 is:",
  "op1": "3· [Ag+] = [PO43-] + [HPO42-] + [H2PO4-] + [H3PO4]",
  "op2": "[Ag+] = 3·([PO43-] + [HPO42-] + [H2PO4-] + [H3PO4])",
  "op3": "[Ag+] + [H+] = 3·([PO43-] + [HPO42-] + [H2PO4-]) + [OH-]",
  "op4": "[Ag+] = [PO43-]3.",
  "cop": 2,
  "year": 2020,
  "category": "Chemistry",
  "unique_id": "2d2974ec-ba73-4dc1-a103-5343f83f0ab3"
}
```

## Answer / Label Format

- Answer column: `cop`
- Observed dtype: `Value('int64')`
- Distinct values observed: [1, 2, 3, 4]
- Value counts: {1: 1389, 2: 1460, 3: 1487, 4: 1285}
- Interpretation: values are integers, which — combined with the `op1`..`op4` columns present in this dataset — indicates this is a 1-indexed pointer into the option columns (i.e. `cop=1` -> `op1` is correct). This is an inference from the observed data layout, not an assumption from the HF dataset card; verify against the card if precision matters downstream.

Sample raw (question truncated, options, cop) triples for verification:

- row 0: question="In relation to iron metabolism and its control mediated by hepcidin, it is true ...", options={'op1': 'The drop in partial oxygen pressure promotes the activation of the hypoxia-inducible factor (HIF), which increases the expression of hepcidin.', 'op2': 'The increase in serum iron or inflammation stimulates the synthesis of hepcidin in the liver, which negatively regulates the function of ferroportin.', 'op3': 'Hepcidin reduces intestinal iron absorption through the inactivation of the divalent metal transporter 1 (DMT1).', 'op4': 'In hereditary hemochromatosis type 1, mutations in the human hemochromatosis protein (HFE) cause an increase in the production of hepcidin.'}, cop=2
- row 1405: question="With regard to the frequency of advice on tobacco use in the general adult popul...", options={'op1': 'It is not necessary to re-question individuals over 25 years old, who it is noted in the medical history, have never smoked.', 'op2': 'It should be carried out every 7-10 months in smokers, coinciding with the assessment of the rest of the factors.', 'op3': 'This should be carried out in all cases whenever the patient comes for a consultation for any other reason.', 'op4': 'It should be performed at least every 3 years in smokers, coinciding with the assessment of the rest of the factors.'}, cop=1
- row 2810: question="Indicate which of the following structures can never be observed in the umbilica...", options={'op1': 'Mesenchymal cells.', 'op2': 'Artery.', 'op3': 'Trophoblast.', 'op4': 'Vein.'}, cop=3

## Options Format

- Option columns found: ['op1', 'op2', 'op3', 'op4']
- Storage: separate flat string columns (one column per option), NOT a list/array column.
- Count of option columns present in schema: 4 (fixed by schema: every row has these same 4 columns)
- Empty/null values per option column: {'op1': 0, 'op2': 0, 'op3': 0, 'op4': 0}
- All 5621 rows have all 4 options populated (non-empty) -> effectively a FIXED count of 4 options per question.

## Category / Specialty Metadata

- Category column found: `category`
- Number of distinct categories: 6

| category | count |
|---|---|
| Pharmacology | 969 |
| Biology | 966 |
| Psychology | 962 |
| Chemistry | 944 |
| Nursing | 923 |
| Medicine | 857 |

## Missing Values

| column | n_null_or_empty |
|---|---|
| `question` | 0 |
| `op1` | 0 |
| `op2` | 0 |
| `op3` | 0 |
| `op4` | 0 |
| `cop` | 0 |

- `cop` values outside expected range [1, 4]: 0

- Total missing/empty cells across checked columns: 0

## Exact-Duplicate Questions (normalized: lowercase, strip whitespace/punctuation)

- Distinct normalized questions: 5569
- Total rows: 5621
- Duplicate groups (normalized question appears >1 time): 38
- Total rows involved in duplicate groups: 90
- "Extra" rows beyond one-per-group (i.e. n_rows - n_distinct_normalized): 52

### Example duplicate pairs (up to 2 groups shown)

**Duplicate group (normalized: "duchenne muscular dystrophy...")** — row indices [852, 2630]

- row 852 raw record:
```json
{
  "exam_id": 181,
  "question": "Duchenne Muscular Dystrophy:",
  "op1": "It is an idiopathic myopathy.",
  "op2": "It affects men and women equally.",
  "op3": "Currently, they are undergoing a curative treatment based on the administration of glucocorticoids.",
  "op4": "It is a recessive hereditary disease linked to the X chromosome.",
  "cop": 4,
  "year": 2020,
  "category": "Medicine",
  "unique_id": "dad16bf8-663d-4b9d-8e29-da568386a585"
}
```
- row 2630 raw record:
```json
{
  "exam_id": 63,
  "question": "Duchenne Muscular Dystrophy:",
  "op1": "It is a recessive disease linked to the Y chromosome.",
  "op2": "It is caused by a deficiency in the dystrophin protein.",
  "op3": "It causes the muscular tone and reflexes to not be maintained.",
  "op4": "It is a less severe variant of Becker's disease.",
  "cop": 2,
  "year": 2020,
  "category": "Pharmacology",
  "unique_id": "5d169b81-788f-4020-9658-873633417b42"
}
```
**Duplicate group (normalized: "indicate the correct answer...")** — row indices [1099, 2641, 2643]

- row 1099 raw record:
```json
{
  "exam_id": 70,
  "question": "Indicate the correct answer:",
  "op1": "Sleep disorders are common in people with type 2 diabetes and cause alterations in the amount, quality, and timing of sleep, associating with a higher risk of obesity and alterations in daytime functioning and glucose metabolism.",
  "op2": "Obstructive sleep apnea affects less than half of the people with type 2 diabetes, and its severity is not associated with blood glucose levels.",
  "op3": "Long sleep durations (> 8 h) do not have negative impacts on diabetes.",
  "op4": "The concept of \"catch-up\" sleep on the weekend alone is sufficient to reverse the impact of insufficient sleep in people with type 2 diabetes.",
  "cop": 1,
  "year": 2023,
  "category": "Nursing",
  "unique_id": "24edd2ec-c786-49d8-a9d3-de0effb58bbd"
}
```
- row 2641 raw record:
```json
{
  "exam_id": 74,
  "question": "Indicate the correct answer:",
  "op1": "Gemfibrozil prevents intestinal absorption of cholesterol by inhibiting the NPC1L1 transporter protein.",
  "op2": "Ezetimibe increases the beta-oxidation of fatty acids and the synthesis of LPL.",
  "op3": "The pleiotropic effects of statins are derived from the reduction of cholesterol.",
  "op4": "Bile acid binding resins decrease the absorption of some drugs such as folic acid.",
  "cop": 4,
  "year": 2020,
  "category": "Pharmacology",
  "unique_id": "e12548d3-05f3-404b-aa9d-9a9c33046138"
}
```

## Explicit `id` Column Check

UNCERTAIN / NOT PRESENT: no column literally named one of ['id', 'ID', 'Id'] exists in the schema. Actual columns: ['exam_id', 'question', 'op1', 'op2', 'op3', 'op4', 'cop', 'year', 'category', 'unique_id']. Note: the dataset DOES contain `unique_id` and `exam_id` columns, which may serve an identifier-like purpose, but neither is literally named `id`. Reporting separately below rather than treating them as confirmed row identifiers.
- `unique_id` present: dtype=`Value('string')`, n_unique=5621 out of 5621 rows (unique per row).
- `exam_id` present: dtype=`Value('int64')`, n_unique=210 out of 5621 rows (NOT unique per row — duplicates exist).

## Qualitative Note on Phrasing/Style (for overlap-analysis teammate)

This script does not perform cross-dataset overlap analysis (that is a separate teammate's task and out of scope here). However, from the 5 representative examples printed above, reviewers should visually check whether question stems resemble USMLE/MedQA-style vignette phrasing (e.g. long clinical-vignette stems with patient age/sex/presenting-symptoms framing) versus CareQA's documented origin as Spanish MIR (Médico Interno Residente) exam questions translated to English. Any structural similarity should be flagged qualitatively by a human reviewer reading the printed examples in this file — no automated similarity score was computed here.