# Cross-Split Drug-Polarity Audit Table

Audit verifying polarity consistency across train and held-out (near-OOD and far-OOD) splits.
Constraint: FAIL if any drug is 'avoid' in train and 'correct' in held-out, or vice versa.

| Rule ID | Family | Split | Split Group | Indication | Drug Entity | Polarity | Action Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `renal_01` | renal | `train` | **train** | Type 2 diabetes mellitus | `metformin` | **AVOID** | Prescribe metformin 1000 mg orally twice daily for 30 days.... |
| `renal_01` | renal | `train` | **train** | Type 2 diabetes mellitus | `insulin` | **CORRECT** | Prescribe glipizide 5 mg orally twice daily for 30 days.... |
| `renal_03` | renal | `train` | **train** | Acute uncomplicated lower urinary tract infection | `nitrofurantoin` | **AVOID** | Prescribe nitrofurantoin 100 mg orally twice daily for 5 day... |
| `renal_03` | renal | `train` | **train** | Acute uncomplicated lower urinary tract infection | `fosfomycin` | **CORRECT** | Prescribe fosfomycin 3 g orally once as a single dose for 1 ... |
| `renal_04` | renal | `train` | **train** | Postherpetic neuralgia / neuropathic pain | `gabapentin` | **AVOID** | Prescribe gabapentin 600 mg orally three times daily for 30 ... |
| `renal_04` | renal | `train` | **train** | Postherpetic neuralgia / neuropathic pain | `gabapentin_dose_reduced` | **CORRECT** | Prescribe gabapentin 300 mg orally twice daily for 30 days.... |
| `preg_01` | pregnancy | `train` | **train** | Essential hypertension | `lisinopril` | **AVOID** | Prescribe lisinopril 10 mg orally once daily for 30 days.... |
| `preg_01` | pregnancy | `train` | **train** | Essential hypertension | `labetalol` | **CORRECT** | Prescribe labetalol 100 mg orally twice daily for 30 days.... |
| `preg_01` | pregnancy | `train` | **train** | Essential hypertension | `nifedipine` | **CORRECT** | Prescribe labetalol 100 mg orally twice daily for 30 days.... |
| `preg_03` | pregnancy | `train` | **train** | Early localized Lyme disease | `doxycycline` | **AVOID** | Prescribe doxycycline hyclate 100 mg orally twice daily for ... |
| `preg_03` | pregnancy | `train` | **train** | Early localized Lyme disease | `cefuroxime_axetil` | **CORRECT** | Prescribe cefuroxime axetil 500 mg orally twice daily for 14... |
| `renal_02` | renal | `near_ood` | **held_out** | Acute deep vein thrombosis / pulmonary embolism anticoagulation | `enoxaparin` | **AVOID** | Prescribe enoxaparin 1 mg/kg subcutaneously every 12 hours f... |
| `renal_02` | renal | `near_ood` | **held_out** | Acute deep vein thrombosis / pulmonary embolism anticoagulation | `enoxaparin_dose_reduced` | **CORRECT** | Prescribe enoxaparin 1 mg/kg subcutaneously once every 24 ho... |
| `renal_02` | renal | `near_ood` | **held_out** | Acute deep vein thrombosis / pulmonary embolism anticoagulation | `unfractionated_heparin` | **CORRECT** | Prescribe enoxaparin 1 mg/kg subcutaneously once every 24 ho... |
| `preg_02` | pregnancy | `near_ood` | **held_out** | Active rheumatoid arthritis | `methotrexate` | **AVOID** | Prescribe oral methotrexate 15 mg once weekly for 30 days.... |
| `preg_02` | pregnancy | `near_ood` | **held_out** | Active rheumatoid arthritis | `hydroxychloroquine` | **CORRECT** | Prescribe hydroxychloroquine 200 mg orally twice daily for 3... |
| `preg_02` | pregnancy | `near_ood` | **held_out** | Active rheumatoid arthritis | `sulfasalazine` | **CORRECT** | Prescribe hydroxychloroquine 200 mg orally twice daily for 3... |
| `allergy_01` | allergy | `far_ood` | **held_out** | Group A streptococcal pharyngitis | `amoxicillin` | **AVOID** | Prescribe amoxicillin 500 mg orally twice daily for 10 days.... |
| `allergy_01` | allergy | `far_ood` | **held_out** | Group A streptococcal pharyngitis | `penicillin` | **AVOID** | Prescribe amoxicillin 500 mg orally twice daily for 10 days.... |
| `allergy_01` | allergy | `far_ood` | **held_out** | Group A streptococcal pharyngitis | `azithromycin` | **CORRECT** | Prescribe azithromycin 500 mg orally once daily for 5 days.... |
| `allergy_02` | allergy | `far_ood` | **held_out** | Acute uncomplicated cystitis | `trimethoprim_sulfamethoxazole` | **AVOID** | Prescribe TMP-SMX 160/800 mg orally twice daily for 3 days.... |
| `allergy_02` | allergy | `far_ood` | **held_out** | Acute uncomplicated cystitis | `sulfonamide` | **AVOID** | Prescribe TMP-SMX 160/800 mg orally twice daily for 3 days.... |
| `allergy_02` | allergy | `far_ood` | **held_out** | Acute uncomplicated cystitis | `ciprofloxacin` | **CORRECT** | Prescribe ciprofloxacin 250 mg orally twice daily for 3 days... |
| `allergy_03` | allergy | `far_ood` | **held_out** | Primary dysmenorrhea | `ibuprofen` | **AVOID** | Prescribe ibuprofen 600 mg orally every 8 hours as needed fo... |
| `allergy_03` | allergy | `far_ood` | **held_out** | Primary dysmenorrhea | `nsaids` | **AVOID** | Prescribe ibuprofen 600 mg orally every 8 hours as needed fo... |
| `allergy_03` | allergy | `far_ood` | **held_out** | Primary dysmenorrhea | `aspirin` | **AVOID** | Prescribe ibuprofen 600 mg orally every 8 hours as needed fo... |
| `allergy_03` | allergy | `far_ood` | **held_out** | Primary dysmenorrhea | `acetaminophen` | **CORRECT** | Prescribe acetaminophen 650 mg orally every 6 hours as neede... |

## Conflict Audit Results
**STATUS: PASS (0 conflicts across train and held-out splits)**
- No drug labeled 'avoid' in train appears as 'correct' in held-out.
- No drug labeled 'correct' in train appears as 'avoid' in held-out.
- Total unique drug entities audited: 27
