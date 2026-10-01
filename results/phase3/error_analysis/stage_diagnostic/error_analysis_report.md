# MedPerturb Phase 3 — Qualitative & Quantitative Error Analysis

- **Total Evaluated Pairs**: 1449
- **Cases where NS Rescues (Base ✗ → Corr ✗ → NS ✓)**: 10
- **Cases where NS Hurts (Base/Corr ✓ → NS ✗)**: 12
- **Net Semantic Benefit**: -2 pairs

## Failure Taxonomy Across Policies

|                               |   Base |   Correctness |   Neuro-Symbolic |
|:------------------------------|-------:|--------------:|-----------------:|
| knowledge_error               |    335 |           341 |              335 |
| evaluation_ambiguity          |    188 |           189 |              187 |
| required_adaptation_failure   |    177 |           177 |              177 |
| representation_instability    |    107 |           101 |              103 |
| instruction_following_failure |     36 |            41 |               37 |
| invalid_option                |     10 |            10 |                9 |
| max_tokens_truncation         |      5 |             5 |                6 |
| parse_failure                 |      4 |             2 |                6 |

## Exemplar Cases: Neuro-Symbolic Rescues

### Question `medmcqa::20929b62-1d50-4e64-9370-8d4966ccb2ae` (no_symbols)
- **Gold Target**: Carotico-cavernous fistula
- **Base Prediction**: `"Fracture sphenoid bone"` (Incorrect)
- **Correctness Prediction**: `"Fracture sphenoid bone"` (Incorrect)
- **Neuro-Symbolic Prediction**: `"Carotico-cavernous fistula"` (CORRECT)

### Question `medqa::0056` (fixed_pos)
- **Gold Target**: Gynecomastia
- **Base Prediction**: `"A"` (Incorrect)
- **Correctness Prediction**: `"A"` (Incorrect)
- **Neuro-Symbolic Prediction**: `"D"` (CORRECT)

### Question `medqa::0167` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["A", "C", "D"]` (Incorrect)
- **Correctness Prediction**: `["A", "C", "D"]` (Incorrect)
- **Neuro-Symbolic Prediction**: `["A", "B", "D"]` (CORRECT)

### Question `medqa::0214` (mcq)
- **Gold Target**: Factitious disorder
- **Base Prediction**: `"A"` (Incorrect)
- **Correctness Prediction**: `"A"` (Incorrect)
- **Neuro-Symbolic Prediction**: `"B"` (CORRECT)

### Question `medqa::0656` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["D"]` (Incorrect)
- **Correctness Prediction**: `["D"]` (Incorrect)
- **Neuro-Symbolic Prediction**: `["A", "B", "C"]` (CORRECT)

## Exemplar Cases: Neuro-Symbolic Regressions (Honest Negative Cases)

### Question `medqa::0014` (roman_numeral)
- **Gold Target**: Rotavirus
- **Base Prediction**: `"III"`
- **Correctness Prediction**: `"IV"`
- **Neuro-Symbolic Prediction**: `"IV"` (INCORRECT)

### Question `medqa::0067` (no_symbols)
- **Gold Target**: Transplacental passage of TSH receptor antibodies
- **Base Prediction**: `"Transplacental passage of TSH receptor antibodies"`
- **Correctness Prediction**: `"Transplacental passage of thyroid peroxidase antibodies"`
- **Neuro-Symbolic Prediction**: `"Transplacental passage of thyroid peroxidase antibodies"` (INCORRECT)

### Question `medqa::0182` (roman_numeral)
- **Gold Target**: Glandular tissue enlargement
- **Base Prediction**: `"III"`
- **Correctness Prediction**: `"II"`
- **Neuro-Symbolic Prediction**: `"III"` (INCORRECT)

### Question `medqa::0182` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["A", "C", "D"]`
- **Correctness Prediction**: `["A", "D"]`
- **Neuro-Symbolic Prediction**: `["A", "D"]` (INCORRECT)

### Question `medqa::0340` (incorrect)
- **Gold Target**: nan
- **Base Prediction**: `["A", "B", "C"]`
- **Correctness Prediction**: `["B", "C", "D"]`
- **Neuro-Symbolic Prediction**: `["B", "C", "D"]` (INCORRECT)
