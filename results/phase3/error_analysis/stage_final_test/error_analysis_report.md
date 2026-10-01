# MedPerturb Phase 3 — Qualitative & Quantitative Error Analysis

- **Total Evaluated Pairs**: 2961
- **Cases where NS Rescues (Base ✗ → Corr ✗ → NS ✓)**: 16
- **Cases where NS Hurts (Base/Corr ✓ → NS ✗)**: 30
- **Net Semantic Benefit**: -14 pairs

## Failure Taxonomy Across Policies

|                               |   Base |   Correctness |   Neuro-Symbolic |
|:------------------------------|-------:|--------------:|-----------------:|
| knowledge_error               |    769 |           766 |              775 |
| evaluation_ambiguity          |    373 |           374 |              373 |
| required_adaptation_failure   |    360 |           362 |              362 |
| representation_instability    |    215 |           207 |              204 |
| instruction_following_failure |     68 |            74 |               70 |
| invalid_option                |     12 |            13 |               14 |
| parse_failure                 |      9 |            10 |               10 |
| max_tokens_truncation         |      5 |             4 |                2 |

## Exemplar Cases: Neuro-Symbolic Rescues

### Question `medmcqa::2fd85795-3aca-4bd2-b367-256d33655e91` (fixed_pos)
- **Gold Target**: Space differential between deciduous canine and molar and their succedaneous permanent teeth
- **Base Prediction**: `"A"` (Incorrect)
- **Correctness Prediction**: `"A"` (Incorrect)
- **Neuro-Symbolic Prediction**: `"D"` (CORRECT)

### Question `medmcqa::40fa7aa0-6036-426b-b694-5ff4a82b6dc7` (fixed_pos)
- **Gold Target**: Closed reduction of zygomatic fracture
- **Base Prediction**: `"A"` (Incorrect)
- **Correctness Prediction**: `"A"` (Incorrect)
- **Neuro-Symbolic Prediction**: `"D"` (CORRECT)

### Question `medmcqa::b924fda5-b814-48da-bd8a-4a5c46c292c6` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["A", "B", "C"]` (Incorrect)
- **Correctness Prediction**: `["A", "B", "C"]` (Incorrect)
- **Neuro-Symbolic Prediction**: `["A", "C", "D"]` (CORRECT)

### Question `medmcqa::c1aa8a36-280b-4195-b6eb-7f5b581ded0d` (fixed_pos)
- **Gold Target**: lateral to canine
- **Base Prediction**: `"B"` (Incorrect)
- **Correctness Prediction**: `"B"` (Incorrect)
- **Neuro-Symbolic Prediction**: `"D"` (CORRECT)

### Question `medqa::0065` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["B", "C", "D"]` (Incorrect)
- **Correctness Prediction**: `["B", "C", "D"]` (Incorrect)
- **Neuro-Symbolic Prediction**: `["A", "C", "D"]` (CORRECT)

## Exemplar Cases: Neuro-Symbolic Regressions (Honest Negative Cases)

### Question `medmcqa::18d1c316-555c-4528-a9c4-e8ce1a613179` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["B", "C", "D"]`
- **Correctness Prediction**: `["A", "B", "C"]`
- **Neuro-Symbolic Prediction**: `["A", "B", "C"]` (INCORRECT)

### Question `medmcqa::1a4b278a-1f53-4306-ba5e-7b5e303120be` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["A", "C", "D"]`
- **Correctness Prediction**: `["A", "C", "D"]`
- **Neuro-Symbolic Prediction**: `["A", "B", "C"]` (INCORRECT)

### Question `medmcqa::309b2357-11a3-4193-ac6c-4ead19255369` (fixed_pos)
- **Gold Target**: Cavity preparation
- **Base Prediction**: `"D"`
- **Correctness Prediction**: `"D"`
- **Neuro-Symbolic Prediction**: `"B"` (INCORRECT)

### Question `medmcqa::4127528f-2cc3-44bc-b07e-446577f5018c` (mcq)
- **Gold Target**: 0.5 ml in 1:1000
- **Base Prediction**: `"A"`
- **Correctness Prediction**: `"C"`
- **Neuro-Symbolic Prediction**: `"C"` (INCORRECT)

### Question `medmcqa::5c55ed82-c9c5-4f5d-8d0e-9cd76cfbeb74` (fixed_pos)
- **Gold Target**: USG for fetal cardiac activity
- **Base Prediction**: `"D"`
- **Correctness Prediction**: `"D"`
- **Neuro-Symbolic Prediction**: `"A"` (INCORRECT)
