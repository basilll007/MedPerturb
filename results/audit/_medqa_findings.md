# MedQA Forensic Audit Findings

Source: `F:\NAACL 2027\data\raw\medqa` (loaded via `datasets.load_from_disk`). Originally from HF dataset `premmahadik05/medqa`, config `questions`.

Phase 0 audit only. All numbers below are computed directly from the local on-disk data by this script; no statistic is fabricated. Anything not directly verifiable is marked `UNCERTAIN: <reason>`.

## Splits found

`['train', 'validation', 'test']`

## 1. Per-split schema, dtypes, and example counts

### Split: `train`  (n_examples = 10178)

Columns / dtypes (as reported by `datasets` Features):

  - `question`: `Value('string')`
  - `answer`: `Value('string')`
  - `options`: `{'A': Value('string'), 'B': Value('string'), 'C': Value('string'), 'D': Value('string')}`
  - `meta_info`: `Value('string')`
  - `answer_idx`: `Value('string')`
  - `metamap_phrases`: `List(Value('string'))`

### Split: `validation`  (n_examples = 1272)

Columns / dtypes (as reported by `datasets` Features):

  - `question`: `Value('string')`
  - `answer`: `Value('string')`
  - `options`: `{'A': Value('string'), 'B': Value('string'), 'C': Value('string'), 'D': Value('string')}`
  - `meta_info`: `Value('string')`
  - `answer_idx`: `Value('string')`
  - `metamap_phrases`: `List(Value('string'))`

### Split: `test`  (n_examples = 1273)

Columns / dtypes (as reported by `datasets` Features):

  - `question`: `Value('string')`
  - `answer`: `Value('string')`
  - `options`: `{'A': Value('string'), 'B': Value('string'), 'C': Value('string'), 'D': Value('string')}`
  - `meta_info`: `Value('string')`
  - `answer_idx`: `Value('string')`
  - `metamap_phrases`: `List(Value('string'))`

**Total examples across all splits: 12723**

### Coarse size signal vs. ReMedQA `medqa_*` family (n=1259 per split)

- `train`: n=10178 -> Superset-sized vs. 1259 (larger by 8919).
- `validation`: n=1272 -> Superset-sized vs. 1259 (larger by 13).
- `test`: n=1273 -> Superset-sized vs. 1259 (larger by 14).

(Raw counts only, reported for the overlap teammate's Task 5 analysis. No conclusion about shared provenance is drawn here.)

## 2. Representative full examples (raw, unmodified)

### `train` — 5 example records

**index 0:**

```json
{
  "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
  "answer": "Nitrofurantoin",
  "options": {
    "A": "Ampicillin",
    "B": "Ceftriaxone",
    "C": "Doxycycline",
    "D": "Nitrofurantoin"
  },
  "meta_info": "step2&3",
  "answer_idx": "D",
  "metamap_phrases": [
    "23 year old pregnant woman",
    "weeks presents",
    "burning",
    "urination",
    "states",
    "started 1 day",
    "worsening",
    "drinking",
    "water",
    "taking cranberry extract",
    "feels well",
    "followed by",
    "doctor",
    "pregnancy",
    "temperature",
    "97",
    "36",
    "blood pressure",
    "mmHg",
    "pulse",
    "80 min",
    "respirations",
    "min",
    "oxygen saturation",
    "98",
    "room air",
    "Physical exam",
    "notable",
    "absence",
    "costovertebral angle tenderness",
    "gravid uterus",
    "following",
    "best treatment",
    "patient"
  ]
}
```

**index 2035:**

```json
{
  "question": "A 45-year-old man is brought into the emergency department after he was hit by a car. The patient was intoxicated and walked into oncoming traffic. He is currently unconscious and has a Glasgow coma scale score of 3. The patient has been admitted multiple times for alcohol intoxication and pancreatitis. The patient is resuscitated with fluid and blood products. An initial trauma survey reveals minor scrapes and abrasions and pelvic instability. The patient’s pelvis is placed in a binder. After further resuscitation the patient becomes responsive and states he is in pain. He is given medications and further resuscitation ensues. One hour later, the patient complains of numbness surrounding his mouth and in his extremities. Which of the following is the most likely explanation of this patient’s current symptoms?",
  "answer": "Transfusion complication",
  "options": {
    "A": "Hypokalemia",
    "B": "Medication complication",
    "C": "Transfusion complication",
    "D": "Trauma to the spinal cord"
  },
  "meta_info": "step2&3",
  "answer_idx": "C",
  "metamap_phrases": [
    "year old man",
    "brought",
    "emergency department",
    "hit by",
    "car",
    "patient",
    "walked",
    "traffic",
    "currently unconscious",
    "Glasgow coma scale score",
    "3",
    "patient",
    "admitted multiple times",
    "alcohol intoxication",
    "pancreatitis",
    "patient",
    "resuscitated",
    "fluid",
    "blood products",
    "initial trauma survey reveals minor scrapes",
    "abrasions",
    "pelvic instability",
    "patients pelvis",
    "placed",
    "binder",
    "further resuscitation",
    "patient",
    "responsive",
    "states",
    "pain",
    "given medications",
    "further resuscitation",
    "One hour later",
    "patient",
    "numbness surrounding",
    "mouth",
    "extremities",
    "following",
    "most likely explanation",
    "patients current symptoms"
  ]
}
```

**index 4071:**

```json
{
  "question": "A 67-year-old male presents to the emergency department with sudden onset shortness of breath and epigastric pain. The patient has a past medical history of GERD, obesity, diabetes mellitus type II, anxiety, glaucoma, and irritable bowel syndrome. His current medications include omeprazole, insulin, metformin, lisinopril, and clonazepam as needed. The patient's temperature is 99.5°F (37.5°C), pulse is 112/min, blood pressure is 90/70 mmHg, respirations are 18/min, and oxygen saturation is 95% on room air. On physical exam the patient's lungs are clear to auscultation bilaterally. JVD is notable and cardiac auscultation is not revealing. An EKG is obtained in the emergency department. The patient is given a bolus of fluids and his pulse becomes 80/min with a blood pressure of 105/75 mmHg. The patient is then started on beta-blockers, oxygen, nitroglycerin, morphine, IV fluids, and aspirin. Repeat vitals demonstrate a blood pressure of 80/65 mmHg. Which of the following is the best explanation of this patient's current vital signs?",
  "answer": "Increased cGMP",
  "options": {
    "A": "Beta-adrenergic blockade",
    "B": "Increased cGMP",
    "C": "Fluid overload",
    "D": "Left ventricular failure"
  },
  "meta_info": "step2&3",
  "answer_idx": "B",
  "metamap_phrases": [
    "67 year old male presents",
    "emergency department",
    "sudden onset shortness of breath",
    "epigastric pain",
    "patient",
    "past medical GERD",
    "obesity",
    "diabetes mellitus type II",
    "anxiety",
    "glaucoma",
    "irritable bowel syndrome",
    "current medications include omeprazole",
    "insulin",
    "metformin",
    "lisinopril",
    "clonazepam as needed",
    "patient's temperature",
    "99",
    "pulse",
    "min",
    "blood pressure",
    "90 70 mmHg",
    "respirations",
    "min",
    "oxygen saturation",
    "95",
    "room air",
    "physical exam",
    "patient's lungs",
    "clear",
    "auscultation",
    "JVD",
    "notable",
    "cardiac auscultation",
    "not revealing",
    "EKG",
    "obtained",
    "emergency department",
    "patient",
    "given",
    "bolus",
    "fluids",
    "pulse",
    "80 min",
    "blood",
    "75 mmHg",
    "patient",
    "then started",
    "beta-blockers",
    "oxygen",
    "nitroglycerin",
    "morphine",
    "IV fluids",
    "aspirin",
    "Repeat",
    "blood pressure",
    "80 65 mmHg",
    "following",
    "best explanation",
    "patient's current vital signs"
  ]
}
```

**index 6106:**

```json
{
  "question": "A 74-year-old woman presents with severe and progressively worsening shortness of breath. She says that her breathing has been difficult for many years but now it is troubling her a lot. She reports a 50-pack-year smoking history and drinks at least 2 alcoholic beverages daily. On physical examination, the patient is leaning forward in her seat and breathing with pursed lips. Which of the following mechanisms best explains the benefit of oxygen supplementation in this patient?",
  "answer": "Increased oxygen diffusion into capillary",
  "options": {
    "A": "Better binding of oxygen to hemoglobin",
    "B": "Decreases respiratory rate and work of breathing",
    "C": "Free radical formation killing pathogens",
    "D": "Increased oxygen diffusion into capillary"
  },
  "meta_info": "step1",
  "answer_idx": "D",
  "metamap_phrases": [
    "74 year old woman presents",
    "severe",
    "worsening shortness of breath",
    "breathing",
    "difficult",
    "years",
    "now",
    "lot",
    "reports",
    "50 pack-year smoking history",
    "drinks at least",
    "alcoholic beverages daily",
    "physical examination",
    "patient",
    "forward",
    "breathing",
    "pursed lips",
    "following mechanisms best",
    "benefit",
    "oxygen supplementation",
    "patient"
  ]
}
```

**index 8142:**

```json
{
  "question": "An 11-year-old boy is brought to his pediatrician by his mother after he has complained of worsening left thumb pain for the last two weeks. The mother reports that the patient was previously healthy. Approximately 2 weeks ago, the family cat bit the patient’s thumb. The area around the bite wound then became red, hot, and slightly swollen and never healed. Earlier this week, the patient also started developing fevers that were recorded at home to be as high as 103.6°F. On exam, the patient's temperature is 102.2°F (39.0°C), blood pressure is 112/72 mmHg, pulse is 92/min, and respirations are 14/min. The patient’s left thumb is tender to touch over the proximal phalanx and the interphalangeal joint, but there is no obvious erythema or swelling. A radiograph performed in clinic is concerning for osteomyelitis at the proximal phalanx. Which of the following is the most likely cause of this patient’s condition?",
  "answer": "Pasteurella multocida",
  "options": {
    "A": "Bartonella henselae",
    "B": "Pasteurella multocida",
    "C": "Pseudomonas aeruginosa",
    "D": "Salmonella spp."
  },
  "meta_info": "step1",
  "answer_idx": "B",
  "metamap_phrases": [
    "year old boy",
    "brought",
    "pediatrician",
    "mother",
    "worsening left thumb pain",
    "last two weeks",
    "mother reports",
    "patient",
    "healthy",
    "Approximately",
    "weeks",
    "family cat bit",
    "patients thumb",
    "area",
    "bite wound then",
    "red",
    "hot",
    "slightly swollen",
    "never healed",
    "Earlier",
    "week",
    "patient",
    "started",
    "fevers",
    "recorded at home to",
    "high",
    "exam",
    "patient's temperature",
    "blood pressure",
    "72 mmHg",
    "pulse",
    "min",
    "respirations",
    "min",
    "patients left thumb",
    "tender",
    "touch",
    "proximal phalanx",
    "interphalangeal joint",
    "erythema",
    "swelling",
    "radiograph performed in clinic",
    "concerning",
    "osteomyelitis",
    "proximal phalanx",
    "following",
    "most likely cause",
    "patients condition"
  ]
}
```

### `validation` — 5 example records

**index 0:**

```json
{
  "question": "A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?",
  "answer": "Ceftriaxone",
  "options": {
    "A": "Gentamicin",
    "B": "Ciprofloxacin",
    "C": "Ceftriaxone",
    "D": "Trimethoprim"
  },
  "meta_info": "step1",
  "answer_idx": "C",
  "metamap_phrases": [
    "21-year-old sexually active male",
    "fever",
    "pain",
    "urination",
    "inflammation",
    "pain in the right knee",
    "culture",
    "joint shows",
    "bacteria",
    "not ferment maltose",
    "polysaccharide capsule",
    "physician orders antibiotic therapy",
    "patient",
    "mechanism of action",
    "medication given blocks cell wall synthesis",
    "following",
    "given"
  ]
}
```

**index 254:**

```json
{
  "question": "A 28-year-old man is brought to the emergency department 20 minutes after being involved in a bicycling accident. He complains of severe pain over the front of his right shoulder. He refuses to move his right arm. Physical examination shows supraclavicular swelling and bruising. The shoulder's range of motion is limited by pain. An x-ray of the shoulder shows a fracture of the middle third of the clavicle with complete superior displacement of the medial clavicular segment. Which of the following muscles is responsible for the displacement of this segment?",
  "answer": "Sternocleidomastoid",
  "options": {
    "A": "Trapezius",
    "B": "Subclavius",
    "C": "Pectoralis major",
    "D": "Sternocleidomastoid"
  },
  "meta_info": "step1",
  "answer_idx": "D",
  "metamap_phrases": [
    "year old man",
    "brought",
    "emergency department 20 minutes",
    "involved",
    "bicycling accident",
    "severe pain",
    "front",
    "right",
    "refuses to move",
    "right arm",
    "Physical examination shows supraclavicular swelling",
    "bruising",
    "shoulder's range of motion",
    "limited",
    "pain",
    "x-ray",
    "shoulder shows",
    "fracture of",
    "middle third",
    "clavicle",
    "complete superior",
    "medial clavicular segment",
    "following muscles",
    "responsible",
    "displacement",
    "segment"
  ]
}
```

**index 508:**

```json
{
  "question": "A 34-year-old man comes to the physician for a routine health maintenance examination required for his occupation as a school bus driver. He feels well and healthy. Upon questioning, he reports that he has smoked 2 joints of marijuana every night after work for the past year for recreational purposes. He typically smokes 5 joints or more on weekends. Which of the following responses by the physician is the most appropriate?",
  "answer": "\"\"\"Have you ever experienced a situation in which you wished you smoked less marijuana?\"\"\"",
  "options": {
    "A": "\"\"\"Let me know when you are ready to stop smoking marijuana. We can talk about specific strategies to help you quit at that time.\"\"\"",
    "B": "\"\"\"Have you ever experienced a situation in which you wished you smoked less marijuana?\"\"\"",
    "C": "\"\"\"We have a great program to help patients with substance use disorder issues. Let me give you more information about specific details.\"\"\"",
    "D": "\"\"\"You should stop smoking marijuana because it is not good for your health.\"\"\""
  },
  "meta_info": "step1",
  "answer_idx": "B",
  "metamap_phrases": [
    "year old man",
    "physician",
    "routine health maintenance examination required",
    "occupation",
    "school",
    "feels well",
    "healthy",
    "questioning",
    "reports",
    "smoked 2 joints",
    "marijuana",
    "night",
    "work",
    "past year",
    "recreational purposes",
    "smokes 5 joints",
    "more",
    "weekends",
    "following responses",
    "physician",
    "most appropriate"
  ]
}
```

**index 763:**

```json
{
  "question": "A 55-year-old woman visits her primary care provider for concerns of frequent headaches. She complains of recurrent headaches and involuntary weight loss, which she attributes to a constant pain along the right side of her jaw that occasionally radiates to her right eye. Her past medical history includes diabetes mellitus type 2 and chronic glomerulonephritis resulting in stage II chronic kidney disease. Her mother passed away in her 70s and had been diagnosed with multiple sclerosis at the age of 50. Today, her blood pressure is 135/90 mm Hg, heart rate is 88/min, respiratory rate is 15/min, and temperature is 36.6°C (97.9°F). The right side of her face is painful to palpation from her jaw to the right side of her scalp. Which of the symptoms below is most commonly associated with the patient’s condition?",
  "answer": "Neck stiffness",
  "options": {
    "A": "Limb muscle weakness",
    "B": "Neck stiffness",
    "C": "Diplopia",
    "D": "Shock-like pain in face"
  },
  "meta_info": "step2&3",
  "answer_idx": "B",
  "metamap_phrases": [
    "55 year old woman visits",
    "primary care provider",
    "concerns",
    "frequent headaches",
    "recurrent headaches",
    "involuntary weight loss",
    "attributes",
    "constant pain",
    "right side of",
    "jaw",
    "occasionally radiates",
    "right eye",
    "past medical history includes diabetes mellitus type 2",
    "chronic glomerulonephritis resulting in stage II chronic kidney disease",
    "mother passed",
    "diagnosed",
    "multiple sclerosis",
    "age",
    "50",
    "Today",
    "blood pressure",
    "90 mm Hg",
    "heart rate",
    "88 min",
    "respiratory rate",
    "min",
    "temperature",
    "36",
    "97 9F",
    "right side of",
    "face",
    "painful",
    "palpation",
    "jaw",
    "right side of",
    "scalp",
    "symptoms",
    "most",
    "associated with",
    "patients condition"
  ]
}
```

**index 1017:**

```json
{
  "question": "A 70-year-old Caucasian woman presents with a 2-week history of blood-tinged sputum. Her past medical history is significant for peptic ulcer disease for which she underwent triple-drug therapy. She is a lifetime non-smoker and worked as a teacher before retiring at the age of 60 years. A review of systems is significant for a weight loss of 6.8 kg (15 lb) over the last 5 months. Her vitals include: blood pressure 135/85 mm Hg, temperature 37.7°C (99.9°F), pulse 95/min, and respiratory rate 18/min. Physical examination is unremarkable. A contrast CT scan of the chest shows an irregular mass in the peripheral region of the inferior lobe of the right lung. A CT-guided biopsy is performed and reveals malignant tissue architecture and gland formation with a significant amount of mucus. Which of the following is the most significant risk factor for this patient’s most likely diagnosis?",
  "answer": "Sex",
  "options": {
    "A": "Medications",
    "B": "Occupational history",
    "C": "Race",
    "D": "Sex"
  },
  "meta_info": "step1",
  "answer_idx": "D",
  "metamap_phrases": [
    "70 year old Caucasian woman presents",
    "2-week history",
    "blood-tinged sputum",
    "past medical history",
    "significant",
    "peptic ulcer disease",
    "triple drug therapy",
    "lifetime non-smoker",
    "worked",
    "teacher",
    "retiring",
    "age",
    "60 years",
    "review of systems",
    "significant",
    "weight loss of 6.8 kg",
    "last",
    "months",
    "include",
    "blood pressure",
    "85 mm Hg",
    "temperature",
    "99 9F",
    "pulse 95 min",
    "respiratory rate",
    "min",
    "Physical examination",
    "unremarkable",
    "contrast CT scan of",
    "chest shows",
    "irregular mass",
    "the peripheral region of",
    "inferior lobe",
    "right lung",
    "CT-guided biopsy",
    "performed",
    "reveals malignant tissue architecture",
    "gland formation",
    "significant amount",
    "mucus",
    "following",
    "most risk factor",
    "patients",
    "likely diagnosis"
  ]
}
```

### `test` — 5 example records

**index 0:**

```json
{
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "answer": "Tell the attending that he cannot fail to disclose this mistake",
  "options": {
    "A": "Disclose the error to the patient and put it in the operative report",
    "B": "Tell the attending that he cannot fail to disclose this mistake",
    "C": "Report the physician to the ethics committee",
    "D": "Refuse to dictate the operative report"
  },
  "meta_info": "step1",
  "answer_idx": "B",
  "metamap_phrases": [
    "junior orthopaedic surgery resident",
    "completing",
    "carpal tunnel repair",
    "department chairman",
    "attending physician",
    "case",
    "resident",
    "cuts",
    "flexor tendon",
    "tendon",
    "repaired",
    "complication",
    "attending",
    "resident",
    "patient",
    "fine",
    "need to report",
    "minor complication",
    "not",
    "patient",
    "not",
    "to make",
    "patient worry",
    "resident to leave",
    "complication out",
    "operative report",
    "following",
    "correct next action",
    "resident to take"
  ]
}
```

**index 254:**

```json
{
  "question": "A 62-year-old man presents to the physician because of incomplete healing of a chest wound. He recently had a triple coronary artery bypass graft 3 weeks ago. His past medical history is significant for type 2 diabetes mellitus and hypertension for the past 25 years. Clinical examination shows the presence of wound dehiscence in the lower 3rd of the sternal region. The wound surface shows the presence of dead necrotic tissue with pus. Computed tomography (CT) of the thorax shows a small fluid collection with fat stranding in the perisurgical soft tissues. What is the most appropriate next step in the management of the patient?",
  "answer": "Surgical debridement",
  "options": {
    "A": "Surgical debridement",
    "B": "Negative pressure wound management",
    "C": "Sternal wiring",
    "D": "Sternal fixation"
  },
  "meta_info": "step2&3",
  "answer_idx": "A",
  "metamap_phrases": [
    "62 year old man presents",
    "physician",
    "incomplete healing",
    "chest",
    "recently",
    "triple coronary artery bypass graft",
    "weeks",
    "past medical history",
    "significant",
    "type 2 diabetes mellitus",
    "hypertension",
    "past",
    "years",
    "Clinical examination shows",
    "presence of wound dehiscence",
    "lower 3rd",
    "sternal region",
    "wound surface shows",
    "presence",
    "dead necrotic tissue",
    "pus",
    "Computed tomography",
    "thorax shows",
    "small fluid collection",
    "fat stranding",
    "soft tissues",
    "most appropriate next step",
    "management",
    "patient"
  ]
}
```

**index 509:**

```json
{
  "question": "An 81-year-old man is brought to the clinic by his son to be evaluated for memory issues. The patient’s son says he has difficulty remembering recent events and names. He says the patient’s symptoms have progressively worsened over the last several years but became acutely worse just recently. Also, yesterday, the patient complained that he could not see out of his right eye, but today he can. When asked about these concerns, the patient seems to have no insight into the problem and reports feeling well. His medical history is significant for diabetes mellitus type 2 and hypertension. He had a left basal ganglia hemorrhage 12 years ago and a right middle cerebral artery infarction 4 years ago. Current medications are amlodipine, aspirin, clopidogrel, metformin, sitagliptin, and valsartan. He lives with his son and can feed himself and change his clothes. There is no history of urinary or fecal incontinence. His vitals include: blood pressure 137/82 mm Hg, pulse 78/min, respiratory rate 16/min, temperature 37.0°C (98.6°F). On physical examination, the patient is alert and oriented. He is unable to perform simple arithmetic calculations and the mini-mental status exam is inconclusive. He can write his name and comprehend written instructions. Muscle strength is 4/5 on the right side. The tone is also slightly reduced on the right side with exaggerated reflexes. His gait is hemiparetic. Which of the following is the most likely diagnosis in this patient?",
  "answer": "Vascular dementia",
  "options": {
    "A": "Alzheimer's disease",
    "B": "Lewy body dementia",
    "C": "Normal-pressure hydrocephalus",
    "D": "Vascular dementia"
  },
  "meta_info": "step2&3",
  "answer_idx": "D",
  "metamap_phrases": [
    "81 year old man",
    "brought",
    "clinic",
    "son to",
    "evaluated",
    "memory issues",
    "patients son",
    "difficulty remembering recent events",
    "names",
    "patients symptoms",
    "worsened",
    "years",
    "worse",
    "recently",
    "patient",
    "not see out",
    "right eye",
    "today",
    "concerns",
    "patient",
    "to",
    "insight",
    "problem",
    "reports feeling well",
    "medical history",
    "significant",
    "diabetes mellitus type 2",
    "hypertension",
    "left",
    "years",
    "right middle cerebral artery infarction",
    "years",
    "Current medications",
    "amlodipine",
    "aspirin",
    "clopidogrel",
    "metformin",
    "sitagliptin",
    "valsartan",
    "lives with",
    "son",
    "feed",
    "change",
    "clothes",
    "history",
    "urinary",
    "fecal incontinence",
    "include",
    "blood pressure",
    "mm Hg",
    "pulse",
    "min",
    "respiratory rate",
    "min",
    "temperature",
    "98",
    "physical examination",
    "patient",
    "alert",
    "oriented",
    "unable to perform simple arithmetic calculations",
    "mini-mental status exam",
    "inconclusive",
    "write",
    "name",
    "written instructions",
    "Muscle strength",
    "4/5",
    "right side",
    "tone",
    "slightly reduced",
    "right side",
    "exaggerated reflexes",
    "gait",
    "following",
    "most likely diagnosis",
    "patient"
  ]
}
```

**index 763:**

```json
{
  "question": "A 42-year-old chronic alcoholic man was admitted to the hospital for inappropriate behavior and disturbed memory. He presents with severe retrograde memory loss, confusion, and confabulation. Neurologic examination showed horizontal nystagmus. He also has bilateral pretibial pitting edema and perioral erythema. CT studies of the brain were normal. The duty physician suspects the patient may be vitamin deficient. Which of the following reactions does the deficient vitamin mediate?",
  "answer": "Alpha-Ketoglutarate + NAD+ + CoA <=> Succinyl-CoA + CO2 + NADH",
  "options": {
    "A": "Alpha-Ketoglutarate + NAD+ + CoA <=> Succinyl-CoA + CO2 + NADH",
    "B": "Succinate + FAD (enzyme bound) <=> Fumarate + FADH2",
    "C": "Isocitrate + NAD+ <=> Alpha-Ketoglutarate + CO2 + NADH",
    "D": "Succinyl-CoA + Pi + GDP <=> Succinate + GTP + CoA"
  },
  "meta_info": "step1",
  "answer_idx": "A",
  "metamap_phrases": [
    "year old chronic alcoholic man",
    "admitted",
    "hospital",
    "inappropriate behavior",
    "memory",
    "presents",
    "severe retrograde memory loss",
    "confusion",
    "confabulation",
    "Neurologic examination showed horizontal nystagmus",
    "bilateral",
    "pitting edema",
    "perioral erythema",
    "CT studies",
    "brain",
    "normal",
    "physician suspects",
    "patient",
    "vitamin deficient",
    "following reactions",
    "deficient vitamin mediate"
  ]
}
```

**index 1018:**

```json
{
  "question": "A 31-year-old man with a history of schizophrenia is brought to the emergency department by police after being found agitated and attempting to steal from a grocery store. His past medical history is only notable for a recent office note from his primary care doctor for treatment of seasonal allergies. His temperature is 101°F (38.3°C), blood pressure is 173/97 mmHg, pulse is 105/min, respirations are 16/min, and oxygen saturation is 98% on room air. Physical exam is notable for a man who is very irritable and restless. He is not cooperative with exam or history and becomes combative requiring intramuscular medications and security restraining him. After this event, the rest of his exam is notable for 7 mm pupils which are equal and reactive to light, spontaneous movement of all limbs, normal sensation, and warm and sweaty skin. The patient is answering questions and states he wants to kill himself. Which of the following substances was most likely used by this patient?",
  "answer": "Cocaine",
  "options": {
    "A": "Alcohol",
    "B": "Cocaine",
    "C": "Diphenhydramine",
    "D": "Haloperidol"
  },
  "meta_info": "step2&3",
  "answer_idx": "B",
  "metamap_phrases": [
    "31 year old man",
    "history of schizophrenia",
    "brought",
    "emergency department",
    "police",
    "found agitated",
    "attempting",
    "steal",
    "grocery store",
    "past medical history",
    "only notable",
    "recent office note",
    "primary care doctor",
    "treatment",
    "seasonal allergies",
    "temperature",
    "3C",
    "blood pressure",
    "97 mmHg",
    "pulse",
    "min",
    "respirations",
    "min",
    "oxygen saturation",
    "98",
    "room air",
    "Physical exam",
    "notable",
    "man",
    "very irritable",
    "restless",
    "not cooperative",
    "exam",
    "history",
    "combative",
    "intramuscular medications",
    "security restraining",
    "event",
    "rest",
    "exam",
    "notable",
    "mm pupils",
    "equal",
    "reactive to light",
    "spontaneous movement",
    "limbs",
    "normal sensation",
    "warm",
    "sweaty skin",
    "patient",
    "answering questions",
    "states",
    "to kill",
    "following substances",
    "most likely used by",
    "patient"
  ]
}
```

## 3. Answer / label format

### `train`

- `answer` column present. Sample raw values: ["Nitrofurantoin", "Placing the infant in a supine position on a firm mattress while sleeping", "Abnormal migration of ventral pancreatic bud", "Thromboembolism", "Von Willebrand disease", "Scorpion sting", "24-hour urine protein", "Gastric fundus in the thorax", "Digoxin", "Persistent congestion"]
- `answer_idx` column present. Sample raw values: ["D", "A", "A", "A", "D", "C", "D", "A", "D", "D"]
  - Distinct `answer_idx` values and counts: {'D': 2383, 'A': 2584, 'C': 2557, 'B': 2654}

### `validation`

- `answer` column present. Sample raw values: ["Ceftriaxone", "Cyclic vomiting syndrome", "Trazodone", "Obtain a urine analysis and urine culture", "Hypoperfusion", "Iron deficiency", "Proteasomal degradation of ubiquitinated proteins", "Non-exertional heat stroke", "Alpha-ketoglutarate dehydrogenase", "Atenolol"]
- `answer_idx` column present. Sample raw values: ["C", "A", "D", "B", "A", "C", "D", "C", "A", "A"]
  - Distinct `answer_idx` values and counts: {'C': 352, 'A': 330, 'D': 274, 'B': 316}

### `test`

- `answer` column present. Sample raw values: ["Tell the attending that he cannot fail to disclose this mistake", "Cross-linking of DNA", "Cholesterol embolization", "Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar", "Ketotifen eye drops", "Reassurance and continuous monitoring", "Common iliac artery aneurysm", "C... [truncated, full length=409]
- `answer_idx` column present. Sample raw values: ["B", "D", "B", "D", "B", "D", "C", "C", "B", "A"]
  - Distinct `answer_idx` values and counts: {'B': 309, 'D': 265, 'C': 346, 'A': 353}

**Interpretation:** `answer` stores the full option text (free-form string, matches the text of one of the `options` values); `answer_idx` stores the single-letter option key (e.g. `A`/`B`/`C`/`D`). This is verified directly against the sampled raw values above, not assumed.

### Cross-check: does `answer` text match `options[answer_idx]`?

- `train`: 200/200 sampled rows have `answer == options[answer_idx]`.
- `validation`: 200/200 sampled rows have `answer == options[answer_idx]`.
- `test`: 200/200 sampled rows have `answer == options[answer_idx]`.

## 4. Options format

### `train`

- Storage type of `options` field: {'dict': 10178}
- Distinct key-sets seen (dict form) and counts: {('A', 'B', 'C', 'D'): 10178}
- Distinct option-count values and counts: {4: 10178}

### `validation`

- Storage type of `options` field: {'dict': 1272}
- Distinct key-sets seen (dict form) and counts: {('A', 'B', 'C', 'D'): 1272}
- Distinct option-count values and counts: {4: 1272}

### `test`

- Storage type of `options` field: {'dict': 1273}
- Distinct key-sets seen (dict form) and counts: {('A', 'B', 'C', 'D'): 1273}
- Distinct option-count values and counts: {4: 1273}

**Interpretation:** options are stored as a fixed-key dict (`A`/`B`/`C`/`D`) per question, i.e. a fixed 4-way multiple choice format, unless the counts above show otherwise for a given split.

## 5. Explicit id/uid column check

### `train`

- Full column list: ['question', 'answer', 'options', 'meta_info', 'answer_idx', 'metamap_phrases']
- UNCERTAIN: no column matching id/uid/uuid/qid naming pattern found. No explicit identifier column exists in this split; row position (dataset index) is the only positional handle, and `meta_info` / `metamap_phrases` (if present) are NOT identifiers.

### `validation`

- Full column list: ['question', 'answer', 'options', 'meta_info', 'answer_idx', 'metamap_phrases']
- UNCERTAIN: no column matching id/uid/uuid/qid naming pattern found. No explicit identifier column exists in this split; row position (dataset index) is the only positional handle, and `meta_info` / `metamap_phrases` (if present) are NOT identifiers.

### `test`

- Full column list: ['question', 'answer', 'options', 'meta_info', 'answer_idx', 'metamap_phrases']
- UNCERTAIN: no column matching id/uid/uuid/qid naming pattern found. No explicit identifier column exists in this split; row position (dataset index) is the only positional handle, and `meta_info` / `metamap_phrases` (if present) are NOT identifiers.

## 6. Missing value counts (question / options / answer / answer_idx)

### `train` (n=10178)

- `question`: 0 missing/empty/null out of 10178
- `options`: 0 missing/empty/null out of 10178 (missing individual option slots: 0)
- `answer`: 0 missing/empty/null out of 10178
- `answer_idx`: 0 missing/empty/null out of 10178
- `meta_info`: 0 missing/empty/null out of 10178
- `metamap_phrases`: 0 missing/empty/null out of 10178

### `validation` (n=1272)

- `question`: 0 missing/empty/null out of 1272
- `options`: 0 missing/empty/null out of 1272 (missing individual option slots: 0)
- `answer`: 0 missing/empty/null out of 1272
- `answer_idx`: 0 missing/empty/null out of 1272
- `meta_info`: 0 missing/empty/null out of 1272
- `metamap_phrases`: 0 missing/empty/null out of 1272

### `test` (n=1273)

- `question`: 0 missing/empty/null out of 1273
- `options`: 0 missing/empty/null out of 1273 (missing individual option slots: 0)
- `answer`: 0 missing/empty/null out of 1273
- `answer_idx`: 0 missing/empty/null out of 1273
- `meta_info`: 0 missing/empty/null out of 1273
- `metamap_phrases`: 0 missing/empty/null out of 1273

## 7. Exact-duplicate question detection (normalized: lowercase, whitespace/punctuation stripped)

### Within-split duplicates

- `train`: 2 distinct question(s) appear more than once, accounting for 4 total rows (of 10178).
  - Example duplicate group (indices [855, 7255]): {"normalized_question": "please refer to the summary above to answer this question this patient is at greatest risk of damage to which of the following cardiovascular structures patient information age 44 years gender m selfidentified ethnicity caucasian site of care office history reason for visitc... [truncated, full length=5202]
- `validation`: 0 distinct question(s) appear more than once, accounting for 0 total rows (of 1272).
- `test`: 0 distinct question(s) appear more than once, accounting for 0 total rows (of 1273).

### Cross-split duplicates (train/validation/test overlap)

- `train` vs `validation`: 0 distinct normalized question(s) appear in BOTH splits (0 row(s) in `train`, 0 row(s) in `validation`).
- `train` vs `test`: 0 distinct normalized question(s) appear in BOTH splits (0 row(s) in `train`, 0 row(s) in `test`).
- `validation` vs `test`: 0 distinct normalized question(s) appear in BOTH splits (0 row(s) in `validation`, 0 row(s) in `test`).
