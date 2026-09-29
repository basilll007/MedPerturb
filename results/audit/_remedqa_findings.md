# ReMedQA Dataset Forensic Audit

Source data: `F:\NAACL 2027\data\raw\remedqa`

This report is generated programmatically by `scripts/_audit_remedqa.py` from the actual on-disk dataset. All numbers below are computed at run time, not fabricated. Any ambiguous finding is marked `UNCERTAIN: <reason>`.

Expected splits (source x perturbation, 3x7): 21
Actual splits found: 21

## 1. Split Structure, Columns, and Representative Examples

Total splits: **21**

| split | n_examples | columns |
|---|---|---|
| medqa_mcq | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medqa_open | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medqa_incorrect | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medqa_roman_numeral | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medqa_none_of_the_provided | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medqa_fixed_pos | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medqa_no_symbols | 1259 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_mcq | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_open | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_incorrect | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_roman_numeral | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_none_of_the_provided | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_fixed_pos | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| medmcqa_no_symbols | 1000 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_mcq | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_open | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_incorrect | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_roman_numeral | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_none_of_the_provided | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_fixed_pos | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |
| mmlu_no_symbols | 895 | id:Value('string'), question:Value('string'), options:Value('string'), answer:Value('string'), prompt:Value('string'), prompt_think:Value('string') |

### Split: `medqa_mcq`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "options": "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}",
  "answer": "B",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) Tell the attending that he cannot fail to disclose this mistake\n(C) Report the physician to the ethics committee\n(D) Refuse to dictate the operative report\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) Tell the attending that he cannot fail to disclose this mistake\n(C) Report the physician to the ethics committee\n(D) Refuse to dictate the operative report\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `B`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
  "options": "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) Cross-linking of DNA\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) Cross-linking of DNA\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

### Split: `medqa_open`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. What is the correct next action for the resident to take?",
  "options": "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}",
  "answer": "Tell the attending that he cannot fail to disclose this mistake",
  "prompt": "The following are open-ended questions about medical knowledge.\nSolve them in a step-by-step fashion, starting by summarizing the available information.\nOutput a single, concise final answer (not a letter).\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. What is the correct next action for the resident to take?\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the concise answer)",
  "prompt_think": "You are given an open-ended question about medical knowledge. Answer by returning a concise answer. \nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. What is the correct next action for the resident to take?\"\n\nAfter you have finished your thinking process, please show your response with **only** the concise final answer, e.g., \"Final Answer: <your concise answer>\".\n"
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `Tell the attending that he cannot fail to disclose this mistake`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which action?",
  "options": "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}",
  "answer": "Cross-linking of DNA",
  "prompt": "The following are open-ended questions about medical knowledge.\nSolve them in a step-by-step fashion, starting by summarizing the available information.\nOutput a single, concise final answer (not a letter).\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which action?\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the concise answer)",
  "prompt_think": "You are given an open-ended question about medical knowledge. Answer by returning a concise answer. \nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which action?\"\n\nAfter you have finished your thinking process, please show your response with **only** the concise final answer, e.g., \"Final Answer: <your concise answer>\".\n"
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `Cross-linking of DNA`

### Split: `medqa_incorrect`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "options": "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}",
  "answer": "['A', 'C', 'D']",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nAnswer the given question by identifying the three incorrect options. \nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) Tell the attending that he cannot fail to disclose this mistake\n(C) Report the physician to the ethics committee\n(D) Refuse to dictate the operative report\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the list of letters corresponding to the incorrect answers, e.g., \"Final Answer: [A, C, D]\")",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the three incorrect option letters, separated by commas.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) Tell the attending that he cannot fail to disclose this mistake\n(C) Report the physician to the ethics committee\n(D) Refuse to dictate the operative report\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `['A', 'C', 'D']`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
  "options": "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}",
  "answer": "['A', 'B', 'C']",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nAnswer the given question by identifying the three incorrect options. \nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) Cross-linking of DNA\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the list of letters corresponding to the incorrect answers, e.g., \"Final Answer: [A, C, D]\")",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the three incorrect option letters, separated by commas.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) Cross-linking of DNA\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `['A', 'B', 'C']`

### Split: `medqa_roman_numeral`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "options": "{'I': 'Disclose the error to the patient and put it in the operative report', 'II': 'Tell the attending that he cannot fail to disclose this mistake', 'III': 'Report the physician to the ethics committee', 'IV': 'Refuse to dictate the operative report'}",
  "answer": "II",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(I) Disclose the error to the patient and put it in the operative report\n(II) Tell the attending that he cannot fail to disclose this mistake\n(III) Report the physician to the ethics committee\n(IV) Refuse to dictate the operative report\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the roman numeral corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s roman numeral.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(I) Disclose the error to the patient and put it in the operative report\n(II) Tell the attending that he cannot fail to disclose this mistake\n(III) Report the physician to the ethics committee\n(IV) Refuse to dictate the operative report\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'I': 'Disclose the error to the patient and put it in the operative report', 'II': 'Tell the attending that he cannot fail to disclose this mistake', 'III': 'Report the physician to the ethics committee', 'IV': 'Refuse to dictate the operative report'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `II`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
  "options": "{'I': 'Inhibition of proteasome', 'II': 'Hyperstabilization of microtubules', 'III': 'Generation of free radicals', 'IV': 'Cross-linking of DNA'}",
  "answer": "IV",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(I) Inhibition of proteasome\n(II) Hyperstabilization of microtubules\n(III) Generation of free radicals\n(IV) Cross-linking of DNA\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the roman numeral corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s roman numeral.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(I) Inhibition of proteasome\n(II) Hyperstabilization of microtubules\n(III) Generation of free radicals\n(IV) Cross-linking of DNA\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'I': 'Inhibition of proteasome', 'II': 'Hyperstabilization of microtubules', 'III': 'Generation of free radicals', 'IV': 'Cross-linking of DNA'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `IV`

### Split: `medqa_none_of_the_provided`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "options": "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'None of the provided options', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}",
  "answer": "B",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. Select the option \"None of the provided options\" if you think that none of the other options is correct.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) None of the provided options\n(C) Report the physician to the ethics committee\n(D) Refuse to dictate the operative report\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) None of the provided options\n(C) Report the physician to the ethics committee\n(D) Refuse to dictate the operative report\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'None of the provided options', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `B`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
  "options": "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'None of the provided options'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. Select the option \"None of the provided options\" if you think that none of the other options is correct.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) None of the provided options\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) None of the provided options\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'None of the provided options'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

### Split: `medqa_fixed_pos`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "options": "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Refuse to dictate the operative report', 'C': 'Report the physician to the ethics committee', 'D': 'Tell the attending that he cannot fail to disclose this mistake'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) Refuse to dictate the operative report\n(C) Report the physician to the ethics committee\n(D) Tell the attending that he cannot fail to disclose this mistake\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n(A) Disclose the error to the patient and put it in the operative report\n(B) Refuse to dictate the operative report\n(C) Report the physician to the ethics committee\n(D) Tell the attending that he cannot fail to disclose this mistake\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Refuse to dictate the operative report', 'C': 'Report the physician to the ethics committee', 'D': 'Tell the attending that he cannot fail to disclose this mistake'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
  "options": "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) Cross-linking of DNA\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n(A) Inhibition of proteasome\n(B) Hyperstabilization of microtubules\n(C) Generation of free radicals\n(D) Cross-linking of DNA\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

### Split: `medqa_no_symbols`

- n_examples: 1259
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "0000",
  "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
  "options": "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}",
  "answer": "Tell the attending that he cannot fail to disclose this mistake",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n- Disclose the error to the patient and put it in the operative report\n- Tell the attending that he cannot fail to disclose this mistake\n- Report the physician to the ethics committee\n- Refuse to dictate the operative report\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the text of the correct option, without any letter or symbol)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct answer.\nQuestion: \"A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n- Disclose the error to the patient and put it in the operative report\n- Tell the attending that he cannot fail to disclose this mistake\n- Report the physician to the ethics committee\n- Refuse to dictate the operative report\"\n\nAfter you have finished your thinking process, please show your response with *only* the text of the correct answer, without any letter or symbol, e.g., \"Final Answer: <text of the correct answer>\".\n"
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `Tell the attending that he cannot fail to disclose this mistake`

**Raw example [1] (unparsed):**
```json
{
  "id": "0001",
  "question": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
  "options": "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}",
  "answer": "Cross-linking of DNA",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n- Inhibition of proteasome\n- Hyperstabilization of microtubules\n- Generation of free radicals\n- Cross-linking of DNA\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the text of the correct option, without any letter or symbol)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct answer.\nQuestion: \"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\n- Inhibition of proteasome\n- Hyperstabilization of microtubules\n- Generation of free radicals\n- Cross-linking of DNA\"\n\nAfter you have finished your thinking process, please show your response with *only* the text of the correct answer, without any letter or symbol, e.g., \"Final Answer: <text of the correct answer>\".\n"
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `Cross-linking of DNA`

### Split: `medmcqa_mcq`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is",
  "options": "{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}",
  "answer": "C",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n(D) blood screening at this point of time will clear the exact picture\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n(D) blood screening at this point of time will clear the exact picture\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `C`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "Concentration of tropicamide:",
  "options": "{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}",
  "answer": "A",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"Concentration of tropicamide:\n(A) 0.01\n(B) 0.02\n(C) 0.03\n(D) 0.04\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"Concentration of tropicamide:\n(A) 0.01\n(B) 0.02\n(C) 0.03\n(D) 0.04\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `A`

### Split: `medmcqa_open`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "What is the best advice for prenatal diagnosis of Down syndrome in a 17-week pregnant woman who previously had a child with Down syndrome and does not want another affected child?",
  "options": "{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}",
  "answer": "Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not",
  "prompt": "The following are open-ended questions about medical knowledge.\nSolve them in a step-by-step fashion, starting by summarizing the available information.\nOutput a single, concise final answer (not a letter).\nQuestion: \"What is the best advice for prenatal diagnosis of Down syndrome in a 17-week pregnant woman who previously had a child with Down syndrome and does not want another affected child?\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the concise answer)",
  "prompt_think": "You are given an open-ended question about medical knowledge. Answer by returning a concise answer. \nQuestion: \"What is the best advice for prenatal diagnosis of Down syndrome in a 17-week pregnant woman who previously had a child with Down syndrome and does not want another affected child?\"\n\nAfter you have finished your thinking process, please show your response with **only** the concise final answer, e.g., \"Final Answer: <your concise answer>\".\n"
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "What is the typical concentration of tropicamide used in ophthalmic preparations?",
  "options": "{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}",
  "answer": "0.01",
  "prompt": "The following are open-ended questions about medical knowledge.\nSolve them in a step-by-step fashion, starting by summarizing the available information.\nOutput a single, concise final answer (not a letter).\nQuestion: \"What is the typical concentration of tropicamide used in ophthalmic preparations?\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the concise answer)",
  "prompt_think": "You are given an open-ended question about medical knowledge. Answer by returning a concise answer. \nQuestion: \"What is the typical concentration of tropicamide used in ophthalmic preparations?\"\n\nAfter you have finished your thinking process, please show your response with **only** the concise final answer, e.g., \"Final Answer: <your concise answer>\".\n"
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}`
- `answer` observed format: json (float)
- `answer` parsed: `0.01`

### Split: `medmcqa_incorrect`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is",
  "options": "{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}",
  "answer": "['A', 'B', 'D']",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nAnswer the given question by identifying the three incorrect options. \nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n(D) blood screening at this point of time will clear the exact picture\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the list of letters corresponding to the incorrect answers, e.g., \"Final Answer: [A, C, D]\")",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the three incorrect option letters, separated by commas.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n(D) blood screening at this point of time will clear the exact picture\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `['A', 'B', 'D']`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "Concentration of tropicamide:",
  "options": "{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}",
  "answer": "['B', 'C', 'D']",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nAnswer the given question by identifying the three incorrect options. \nQuestion: \"Concentration of tropicamide:\n(A) 0.01\n(B) 0.02\n(C) 0.03\n(D) 0.04\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the list of letters corresponding to the incorrect answers, e.g., \"Final Answer: [A, C, D]\")",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the three incorrect option letters, separated by commas.\nQuestion: \"Concentration of tropicamide:\n(A) 0.01\n(B) 0.02\n(C) 0.03\n(D) 0.04\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `['B', 'C', 'D']`

### Split: `medmcqa_roman_numeral`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is",
  "options": "{'I': 'No test is required now as her age is below 35 years', 'II': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'III': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'IV': 'blood screening at this point of time will clear the exact picture'}",
  "answer": "III",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(I) No test is required now as her age is below 35 years\n(II) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(III) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n(IV) blood screening at this point of time will clear the exact picture\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the roman numeral corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s roman numeral.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(I) No test is required now as her age is below 35 years\n(II) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(III) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n(IV) blood screening at this point of time will clear the exact picture\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'I': 'No test is required now as her age is below 35 years', 'II': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'III': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'IV': 'blood screening at this point of time will clear the exact picture'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `III`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "Concentration of tropicamide:",
  "options": "{'I': '0.01', 'II': '0.02', 'III': '0.03', 'IV': '0.04'}",
  "answer": "I",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"Concentration of tropicamide:\n(I) 0.01\n(II) 0.02\n(III) 0.03\n(IV) 0.04\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the roman numeral corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s roman numeral.\nQuestion: \"Concentration of tropicamide:\n(I) 0.01\n(II) 0.02\n(III) 0.03\n(IV) 0.04\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'I': '0.01', 'II': '0.02', 'III': '0.03', 'IV': '0.04'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `I`

### Split: `medmcqa_none_of_the_provided`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is",
  "options": "{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'None of the provided options', 'D': 'blood screening at this point of time will clear the exact picture'}",
  "answer": "C",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. Select the option \"None of the provided options\" if you think that none of the other options is correct.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) None of the provided options\n(D) blood screening at this point of time will clear the exact picture\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) None of the provided options\n(D) blood screening at this point of time will clear the exact picture\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'None of the provided options', 'D': 'blood screening at this point of time will clear the exact picture'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `C`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "Concentration of tropicamide:",
  "options": "{'A': 'None of the provided options', 'B': '0.02', 'C': '0.03', 'D': '0.04'}",
  "answer": "A",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. Select the option \"None of the provided options\" if you think that none of the other options is correct.\nQuestion: \"Concentration of tropicamide:\n(A) None of the provided options\n(B) 0.02\n(C) 0.03\n(D) 0.04\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"Concentration of tropicamide:\n(A) None of the provided options\n(B) 0.02\n(C) 0.03\n(D) 0.04\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'None of the provided options', 'B': '0.02', 'C': '0.03', 'D': '0.04'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `A`

### Split: `medmcqa_fixed_pos`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is",
  "options": "{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'blood screening at this point of time will clear the exact picture', 'D': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) blood screening at this point of time will clear the exact picture\n(D) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n(A) No test is required now as her age is below 35 years\n(B) Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n(C) blood screening at this point of time will clear the exact picture\n(D) Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'blood screening at this point of time will clear the exact picture', 'D': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "Concentration of tropicamide:",
  "options": "{'A': '0.04', 'B': '0.02', 'C': '0.03', 'D': '0.01'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"Concentration of tropicamide:\n(A) 0.04\n(B) 0.02\n(C) 0.03\n(D) 0.01\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"Concentration of tropicamide:\n(A) 0.04\n(B) 0.02\n(C) 0.03\n(D) 0.01\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': '0.04', 'B': '0.02', 'C': '0.03', 'D': '0.01'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

### Split: `medmcqa_no_symbols`

- n_examples: 1000
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "b64a9cd7-d076-4c55-8be1-f9c44fece6cc",
  "question": "A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is",
  "options": "{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}",
  "answer": "Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n- No test is required now as her age is below 35 years\n- Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n- Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n- blood screening at this point of time will clear the exact picture\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the text of the correct option, without any letter or symbol)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct answer.\nQuestion: \"A 29 yrs old woman with a pregnancy of 17 week has a 10 years old boy with down syndrome. She does not want another down syndrome kid; best advice to her is\n- No test is required now as her age is below 35 years\n- Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not\n- Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not\n- blood screening at this point of time will clear the exact picture\"\n\nAfter you have finished your thinking process, please show your response with *only* the text of the correct answer, without any letter or symbol, e.g., \"Final Answer: <text of the correct answer>\".\n"
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'No test is required now as her age is below 35 years', 'B': 'Ultra sound at this point of time will definitely tell her that next baby will be down syndromic or not', 'C': 'Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not', 'D': 'blood screening at this point of time will clear the exact picture'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not`

**Raw example [1] (unparsed):**
```json
{
  "id": "17360c6c-2c98-4fe2-aa85-487dcf4678df",
  "question": "Concentration of tropicamide:",
  "options": "{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}",
  "answer": "0.01",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"Concentration of tropicamide:\n- 0.01\n- 0.02\n- 0.03\n- 0.04\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the text of the correct option, without any letter or symbol)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct answer.\nQuestion: \"Concentration of tropicamide:\n- 0.01\n- 0.02\n- 0.03\n- 0.04\"\n\nAfter you have finished your thinking process, please show your response with *only* the text of the correct answer, without any letter or symbol, e.g., \"Final Answer: <text of the correct answer>\".\n"
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': '0.01', 'B': '0.02', 'C': '0.03', 'D': '0.04'}`
- `answer` observed format: json (float)
- `answer` parsed: `0.01`

### Split: `mmlu_mcq`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
  "options": "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}",
  "answer": "A",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) paralysis of the facial muscles.\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) paralysis of the facial muscles.\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `A`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "A \"dished face\" profile is often associated with",
  "options": "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}",
  "answer": "B",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) a recessive maxilla due to failure of elongation of the cranial base.\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) defective development of the maxillary air sinus.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) a recessive maxilla due to failure of elongation of the cranial base.\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) defective development of the maxillary air sinus.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `B`

### Split: `mmlu_open`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "What clinical deficit would result from compression of the facial nerve at the stylomastoid foramen on one side?",
  "options": "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}",
  "answer": "paralysis of the facial muscles.",
  "prompt": "The following are open-ended questions about medical knowledge.\nSolve them in a step-by-step fashion, starting by summarizing the available information.\nOutput a single, concise final answer (not a letter).\nQuestion: \"What clinical deficit would result from compression of the facial nerve at the stylomastoid foramen on one side?\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the concise answer)",
  "prompt_think": "You are given an open-ended question about medical knowledge. Answer by returning a concise answer. \nQuestion: \"What clinical deficit would result from compression of the facial nerve at the stylomastoid foramen on one side?\"\n\nAfter you have finished your thinking process, please show your response with **only** the concise final answer, e.g., \"Final Answer: <your concise answer>\".\n"
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `paralysis of the facial muscles.`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "With which craniofacial abnormality is a \"dished face\" profile most often associated?",
  "options": "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}",
  "answer": "a recessive maxilla due to failure of elongation of the cranial base.",
  "prompt": "The following are open-ended questions about medical knowledge.\nSolve them in a step-by-step fashion, starting by summarizing the available information.\nOutput a single, concise final answer (not a letter).\nQuestion: \"With which craniofacial abnormality is a \"dished face\" profile most often associated?\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the concise answer)",
  "prompt_think": "You are given an open-ended question about medical knowledge. Answer by returning a concise answer. \nQuestion: \"With which craniofacial abnormality is a \"dished face\" profile most often associated?\"\n\nAfter you have finished your thinking process, please show your response with **only** the concise final answer, e.g., \"Final Answer: <your concise answer>\".\n"
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `a recessive maxilla due to failure of elongation of the cranial base.`

### Split: `mmlu_incorrect`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
  "options": "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}",
  "answer": "['B', 'C', 'D']",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nAnswer the given question by identifying the three incorrect options. \nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) paralysis of the facial muscles.\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the list of letters corresponding to the incorrect answers, e.g., \"Final Answer: [A, C, D]\")",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the three incorrect option letters, separated by commas.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) paralysis of the facial muscles.\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `['B', 'C', 'D']`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "A \"dished face\" profile is often associated with",
  "options": "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}",
  "answer": "['A', 'C', 'D']",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nAnswer the given question by identifying the three incorrect options. \nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) a recessive maxilla due to failure of elongation of the cranial base.\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) defective development of the maxillary air sinus.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the list of letters corresponding to the incorrect answers, e.g., \"Final Answer: [A, C, D]\")",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the three incorrect option letters, separated by commas.\nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) a recessive maxilla due to failure of elongation of the cranial base.\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) defective development of the maxillary air sinus.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `['A', 'C', 'D']`

### Split: `mmlu_roman_numeral`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
  "options": "{'I': 'paralysis of the facial muscles.', 'II': 'paralysis of the facial muscles and loss of taste.', 'III': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'IV': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}",
  "answer": "I",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(I) paralysis of the facial muscles.\n(II) paralysis of the facial muscles and loss of taste.\n(III) paralysis of the facial muscles, loss of taste and lacrimation.\n(IV) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the roman numeral corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s roman numeral.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(I) paralysis of the facial muscles.\n(II) paralysis of the facial muscles and loss of taste.\n(III) paralysis of the facial muscles, loss of taste and lacrimation.\n(IV) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'I': 'paralysis of the facial muscles.', 'II': 'paralysis of the facial muscles and loss of taste.', 'III': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'IV': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `I`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "A \"dished face\" profile is often associated with",
  "options": "{'I': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'II': 'a recessive maxilla due to failure of elongation of the cranial base.', 'III': 'an enlarged frontal bone due to hydrocephaly.', 'IV': 'defective development of the maxillary air sinus.'}",
  "answer": "II",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. \nQuestion: \"A \"dished face\" profile is often associated with\n(I) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(II) a recessive maxilla due to failure of elongation of the cranial base.\n(III) an enlarged frontal bone due to hydrocephaly.\n(IV) defective development of the maxillary air sinus.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the roman numeral corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s roman numeral.\nQuestion: \"A \"dished face\" profile is often associated with\n(I) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(II) a recessive maxilla due to failure of elongation of the cranial base.\n(III) an enlarged frontal bone due to hydrocephaly.\n(IV) defective development of the maxillary air sinus.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'I': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'II': 'a recessive maxilla due to failure of elongation of the cranial base.', 'III': 'an enlarged frontal bone due to hydrocephaly.', 'IV': 'defective development of the maxillary air sinus.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `II`

### Split: `mmlu_none_of_the_provided`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
  "options": "{'A': 'None of the provided options', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}",
  "answer": "A",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. Select the option \"None of the provided options\" if you think that none of the other options is correct.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) None of the provided options\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) None of the provided options\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'None of the provided options', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `A`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "A \"dished face\" profile is often associated with",
  "options": "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'None of the provided options', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}",
  "answer": "B",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer. Select the option \"None of the provided options\" if you think that none of the other options is correct.\nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) None of the provided options\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) defective development of the maxillary air sinus.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) None of the provided options\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) defective development of the maxillary air sinus.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'None of the provided options', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `B`

### Split: `mmlu_fixed_pos`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
  "options": "{'A': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles.'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n(A) paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\n(B) paralysis of the facial muscles and loss of taste.\n(C) paralysis of the facial muscles, loss of taste and lacrimation.\n(D) paralysis of the facial muscles.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "A \"dished face\" profile is often associated with",
  "options": "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'defective development of the maxillary air sinus.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'a recessive maxilla due to failure of elongation of the cranial base.'}",
  "answer": "D",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) defective development of the maxillary air sinus.\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) a recessive maxilla due to failure of elongation of the cranial base.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct option’s letter.\nQuestion: \"A \"dished face\" profile is often associated with\n(A) a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n(B) defective development of the maxillary air sinus.\n(C) an enlarged frontal bone due to hydrocephaly.\n(D) a recessive maxilla due to failure of elongation of the cranial base.\"\n\nEnclose your final answer within \\boxed{}."
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'defective development of the maxillary air sinus.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'a recessive maxilla due to failure of elongation of the cranial base.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `D`

### Split: `mmlu_no_symbols`

- n_examples: 895
- columns/dtypes: {'id': "Value('string')", 'question': "Value('string')", 'options': "Value('string')", 'answer': "Value('string')", 'prompt': "Value('string')", 'prompt_think': "Value('string')"}

**Raw example [0] (unparsed):**
```json
{
  "id": "anatomy-000",
  "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
  "options": "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}",
  "answer": "paralysis of the facial muscles.",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n- paralysis of the facial muscles.\n- paralysis of the facial muscles and loss of taste.\n- paralysis of the facial muscles, loss of taste and lacrimation.\n- paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the text of the correct option, without any letter or symbol)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct answer.\nQuestion: \"A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral\n- paralysis of the facial muscles.\n- paralysis of the facial muscles and loss of taste.\n- paralysis of the facial muscles, loss of taste and lacrimation.\n- paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.\"\n\nAfter you have finished your thinking process, please show your response with *only* the text of the correct answer, without any letter or symbol, e.g., \"Final Answer: <text of the correct answer>\".\n"
}
```
**Parsed example [0]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `paralysis of the facial muscles.`

**Raw example [1] (unparsed):**
```json
{
  "id": "anatomy-001",
  "question": "A \"dished face\" profile is often associated with",
  "options": "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}",
  "answer": "a recessive maxilla due to failure of elongation of the cranial base.",
  "prompt": "The following are multiple choice questions about medical knowledge. \nSolve them in a step-by-step fashion, starting by summarizing the available information. \nOutput a single option from the four options as the final answer.\nQuestion: \"A \"dished face\" profile is often associated with\n- a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n- a recessive maxilla due to failure of elongation of the cranial base.\n- an enlarged frontal bone due to hydrocephaly.\n- defective development of the maxillary air sinus.\"\n\nResponse (think step by step and then end with \"Final Answer:\" followed by *only* the text of the correct option, without any letter or symbol)",
  "prompt_think": "You are given a multiple choice question about medical knowledge. Answer by returning the correct answer.\nQuestion: \"A \"dished face\" profile is often associated with\n- a protruding mandible due to reactivation of the condylar cartilage by acromegaly.\n- a recessive maxilla due to failure of elongation of the cranial base.\n- an enlarged frontal bone due to hydrocephaly.\n- defective development of the maxillary air sinus.\"\n\nAfter you have finished your thinking process, please show your response with *only* the text of the correct answer, without any letter or symbol, e.g., \"Final Answer: <text of the correct answer>\".\n"
}
```
**Parsed example [1]:**
- `options` observed format: raw-string (not valid JSON)
- `options` parsed: `{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective development of the maxillary air sinus.'}`
- `answer` observed format: raw-string (not valid JSON)
- `answer` parsed: `a recessive maxilla due to failure of elongation of the cranial base.`

## 2. Duplicate `id` Checks Within Each Split

| split | n_rows | n_unique_ids | n_duplicate_id_values |
|---|---|---|---|
| medqa_mcq | 1259 | 1259 | 0 |
| medqa_open | 1259 | 1259 | 0 |
| medqa_incorrect | 1259 | 1259 | 0 |
| medqa_roman_numeral | 1259 | 1259 | 0 |
| medqa_none_of_the_provided | 1259 | 1259 | 0 |
| medqa_fixed_pos | 1259 | 1259 | 0 |
| medqa_no_symbols | 1259 | 1259 | 0 |
| medmcqa_mcq | 1000 | 1000 | 0 |
| medmcqa_open | 1000 | 1000 | 0 |
| medmcqa_incorrect | 1000 | 1000 | 0 |
| medmcqa_roman_numeral | 1000 | 1000 | 0 |
| medmcqa_none_of_the_provided | 1000 | 1000 | 0 |
| medmcqa_fixed_pos | 1000 | 1000 | 0 |
| medmcqa_no_symbols | 1000 | 1000 | 0 |
| mmlu_mcq | 895 | 895 | 0 |
| mmlu_open | 895 | 895 | 0 |
| mmlu_incorrect | 895 | 895 | 0 |
| mmlu_roman_numeral | 895 | 895 | 0 |
| mmlu_none_of_the_provided | 895 | 895 | 0 |
| mmlu_fixed_pos | 895 | 895 | 0 |
| mmlu_no_symbols | 895 | 895 | 0 |

No duplicate `id` values found in any split.

## 3. Cross-Perturbation ID Alignment Per Source Family

### Source: `medqa`

| perturbation | n_ids_in_split |
|---|---|
| mcq | 1259 |
| open | 1259 |
| incorrect | 1259 |
| roman_numeral | 1259 |
| none_of_the_provided | 1259 |
| fixed_pos | 1259 |
| no_symbols | 1259 |

- Union of ids across all 7 perturbations: **1259**
- Intersection (ids common to ALL 7 perturbations): **1259**
- Ids present in union but missing from at least one perturbation: **0**
- Cross-check: ids appearing in all 7 perturbation splits: **1259** (should equal intersection count above: MATCH)
- Ids appearing in only SOME (not all) perturbation splits: **0**

### Source: `medmcqa`

| perturbation | n_ids_in_split |
|---|---|
| mcq | 1000 |
| open | 1000 |
| incorrect | 1000 |
| roman_numeral | 1000 |
| none_of_the_provided | 1000 |
| fixed_pos | 1000 |
| no_symbols | 1000 |

- Union of ids across all 7 perturbations: **1000**
- Intersection (ids common to ALL 7 perturbations): **1000**
- Ids present in union but missing from at least one perturbation: **0**
- Cross-check: ids appearing in all 7 perturbation splits: **1000** (should equal intersection count above: MATCH)
- Ids appearing in only SOME (not all) perturbation splits: **0**

### Source: `mmlu`

| perturbation | n_ids_in_split |
|---|---|
| mcq | 895 |
| open | 895 |
| incorrect | 895 |
| roman_numeral | 895 |
| none_of_the_provided | 895 |
| fixed_pos | 895 |
| no_symbols | 895 |

- Union of ids across all 7 perturbations: **895**
- Intersection (ids common to ALL 7 perturbations): **895**
- Ids present in union but missing from at least one perturbation: **0**
- Cross-check: ids appearing in all 7 perturbation splits: **895** (should equal intersection count above: MATCH)
- Ids appearing in only SOME (not all) perturbation splits: **0**

## 4. Side-by-Side Inspection of Shared IDs Across All 7 Perturbations

### Source: `medqa` — sample of 5 shared ids

#### id = `0000`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committe... | B | "B" |
| open | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committe... | Tell the attending that he cannot fail to disclose this mistake | "Tell the attending that he cannot fail to disclose this mistake" |
| incorrect | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committe... | ['A', 'C', 'D'] | "['A', 'C', 'D']" |
| roman_numeral | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'I': 'Disclose the error to the patient and put it in the operative report', 'II': 'Tell the attending that he cannot fail to disclose this mistake', 'III': 'Report the physician to the ethics commi... | II | "II" |
| none_of_the_provided | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'None of the provided options', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the ope... | B | "B" |
| fixed_pos | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Refuse to dictate the operative report', 'C': 'Report the physician to the ethics committee', 'D': 'Tell the attend... | D | "D" |
| no_symbols | A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. Th... | "{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committe... | Tell the attending that he cannot fail to disclose this mistake | "Tell the attending that he cannot fail to disclose this mistake" |

#### id = `0001`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}" | D | "D" |
| open | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}" | Cross-linking of DNA | "Cross-linking of DNA" |
| incorrect | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}" | ['A', 'B', 'C'] | "['A', 'B', 'C']" |
| roman_numeral | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'I': 'Inhibition of proteasome', 'II': 'Hyperstabilization of microtubules', 'III': 'Generation of free radicals', 'IV': 'Cross-linking of DNA'}" | IV | "IV" |
| none_of_the_provided | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'None of the provided options'}" | D | "D" |
| fixed_pos | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}" | D | "D" |
| no_symbols | A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemot... | "{'A': 'Inhibition of proteasome', 'B': 'Hyperstabilization of microtubules', 'C': 'Generation of free radicals', 'D': 'Cross-linking of DNA'}" | Cross-linking of DNA | "Cross-linking of DNA" |

#### id = `0002`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'A': 'Renal papillary necrosis', 'B': 'Cholesterol embolization', 'C': 'Eosinophilic granulomatosis with polyangiitis', 'D': 'Polyarteritis nodosa'}" | B | "B" |
| open | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'A': 'Renal papillary necrosis', 'B': 'Cholesterol embolization', 'C': 'Eosinophilic granulomatosis with polyangiitis', 'D': 'Polyarteritis nodosa'}" | Cholesterol embolization | "Cholesterol embolization" |
| incorrect | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'A': 'Renal papillary necrosis', 'B': 'Cholesterol embolization', 'C': 'Eosinophilic granulomatosis with polyangiitis', 'D': 'Polyarteritis nodosa'}" | ['A', 'C', 'D'] | "['A', 'C', 'D']" |
| roman_numeral | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'I': 'Renal papillary necrosis', 'II': 'Cholesterol embolization', 'III': 'Eosinophilic granulomatosis with polyangiitis', 'IV': 'Polyarteritis nodosa'}" | II | "II" |
| none_of_the_provided | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'A': 'Renal papillary necrosis', 'B': 'None of the provided options', 'C': 'Eosinophilic granulomatosis with polyangiitis', 'D': 'Polyarteritis nodosa'}" | B | "B" |
| fixed_pos | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'A': 'Renal papillary necrosis', 'B': 'Polyarteritis nodosa', 'C': 'Eosinophilic granulomatosis with polyangiitis', 'D': 'Cholesterol embolization'}" | D | "D" |
| no_symbols | Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus an... | "{'A': 'Renal papillary necrosis', 'B': 'Cholesterol embolization', 'C': 'Eosinophilic granulomatosis with polyangiitis', 'D': 'Polyarteritis nodosa'}" | Cholesterol embolization | "Cholesterol embolization" |

#### id = `0003`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'A': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'B': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal... | D | "D" |
| open | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'A': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'B': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal... | Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar | "Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar" |
| incorrect | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'A': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'B': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal... | ['A', 'B', 'C'] | "['A', 'B', 'C']" |
| roman_numeral | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'I': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'II': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoa... | IV | "IV" |
| none_of_the_provided | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'A': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'B': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal... | D | "D" |
| fixed_pos | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'A': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'B': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal... | D | "D" |
| no_symbols | A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and ... | "{'A': 'Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar', 'B': 'Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal... | Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar | "Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar" |

#### id = `0004`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'A': 'Erythromycin ointment', 'B': 'Ketotifen eye drops', 'C': 'Warm compresses', 'D': 'Fluorometholone eye drops'}" | B | "B" |
| open | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'A': 'Erythromycin ointment', 'B': 'Ketotifen eye drops', 'C': 'Warm compresses', 'D': 'Fluorometholone eye drops'}" | Ketotifen eye drops | "Ketotifen eye drops" |
| incorrect | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'A': 'Erythromycin ointment', 'B': 'Ketotifen eye drops', 'C': 'Warm compresses', 'D': 'Fluorometholone eye drops'}" | ['A', 'C', 'D'] | "['A', 'C', 'D']" |
| roman_numeral | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'I': 'Erythromycin ointment', 'II': 'Ketotifen eye drops', 'III': 'Warm compresses', 'IV': 'Fluorometholone eye drops'}" | II | "II" |
| none_of_the_provided | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'A': 'Erythromycin ointment', 'B': 'None of the provided options', 'C': 'Warm compresses', 'D': 'Fluorometholone eye drops'}" | B | "B" |
| fixed_pos | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'A': 'Erythromycin ointment', 'B': 'Fluorometholone eye drops', 'C': 'Warm compresses', 'D': 'Ketotifen eye drops'}" | D | "D" |
| no_symbols | A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around ... | "{'A': 'Erythromycin ointment', 'B': 'Ketotifen eye drops', 'C': 'Warm compresses', 'D': 'Fluorometholone eye drops'}" | Ketotifen eye drops | "Ketotifen eye drops" |

### Source: `medmcqa` — sample of 5 shared ids

#### id = `0009b2fd-7e72-4ed0-b486-92ccb24e43f3`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Which of the following marker persists in chronic hepatitis and recurrent hepatitis? | "{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}" | A | "A" |
| open | Which serological marker persists in chronic hepatitis and recurrent hepatitis? | "{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}" | IgG Anti HbcAg | "IgG Anti HbcAg" |
| incorrect | Which of the following marker persists in chronic hepatitis and recurrent hepatitis? | "{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}" | ['B', 'C', 'D'] | "['B', 'C', 'D']" |
| roman_numeral | Which of the following marker persists in chronic hepatitis and recurrent hepatitis? | "{'I': 'IgG Anti HbcAg', 'II': 'HBsAg', 'III': 'IgG Anti HBsAG', 'IV': 'Anti Hbs'}" | I | "I" |
| none_of_the_provided | Which of the following marker persists in chronic hepatitis and recurrent hepatitis? | "{'A': 'None of the provided options', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}" | A | "A" |
| fixed_pos | Which of the following marker persists in chronic hepatitis and recurrent hepatitis? | "{'A': 'Anti Hbs', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'IgG Anti HbcAg'}" | D | "D" |
| no_symbols | Which of the following marker persists in chronic hepatitis and recurrent hepatitis? | "{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}" | IgG Anti HbcAg | "IgG Anti HbcAg" |

#### id = `006fea5f-8d1c-489b-9ea6-1028a64484ab`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Fingerprinting (FINDER) involves recording prints of 8 fingers. Which finger pair is excluded? | "{'A': 'Ring finger', 'B': 'Thumb', 'C': 'Little finger', 'D': 'Middle finger'}" | C | "C" |
| open | When recording fingerprints of 8 fingers in the FINDER method, which pair of fingers is excluded? | "{'A': 'Ring finger', 'B': 'Thumb', 'C': 'Little finger', 'D': 'Middle finger'}" | Little finger | "Little finger" |
| incorrect | Fingerprinting (FINDER) involves recording prints of 8 fingers. Which finger pair is excluded? | "{'A': 'Ring finger', 'B': 'Thumb', 'C': 'Little finger', 'D': 'Middle finger'}" | ['A', 'B', 'D'] | "['A', 'B', 'D']" |
| roman_numeral | Fingerprinting (FINDER) involves recording prints of 8 fingers. Which finger pair is excluded? | "{'I': 'Ring finger', 'II': 'Thumb', 'III': 'Little finger', 'IV': 'Middle finger'}" | III | "III" |
| none_of_the_provided | Fingerprinting (FINDER) involves recording prints of 8 fingers. Which finger pair is excluded? | "{'A': 'Ring finger', 'B': 'Thumb', 'C': 'None of the provided options', 'D': 'Middle finger'}" | C | "C" |
| fixed_pos | Fingerprinting (FINDER) involves recording prints of 8 fingers. Which finger pair is excluded? | "{'A': 'Ring finger', 'B': 'Thumb', 'C': 'Middle finger', 'D': 'Little finger'}" | D | "D" |
| no_symbols | Fingerprinting (FINDER) involves recording prints of 8 fingers. Which finger pair is excluded? | "{'A': 'Ring finger', 'B': 'Thumb', 'C': 'Little finger', 'D': 'Middle finger'}" | Little finger | "Little finger" |

#### id = `007d53f1-6364-4482-b62e-5b51134a222c`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A child has received full Rabies vaccination in December 2018 and now presented with oozing wound on Great toe and the pet had vaccination also. Next line of management is | "{'A': 'No vaccine required', 'B': 'RIG + 5 doses of vaccine', 'C': '5 doses of vaccines only', 'D': '2 doses of Rabies vaccine'}" | D | "D" |
| open | A child has received full Rabies vaccination in December 2018 and now presents with an oozing wound on the great toe after exposure to a pet that has also been vaccinated. What is the next line of man... | "{'A': 'No vaccine required', 'B': 'RIG + 5 doses of vaccine', 'C': '5 doses of vaccines only', 'D': '2 doses of Rabies vaccine'}" | 2 doses of Rabies vaccine | "2 doses of Rabies vaccine" |
| incorrect | A child has received full Rabies vaccination in December 2018 and now presented with oozing wound on Great toe and the pet had vaccination also. Next line of management is | "{'A': 'No vaccine required', 'B': 'RIG + 5 doses of vaccine', 'C': '5 doses of vaccines only', 'D': '2 doses of Rabies vaccine'}" | ['A', 'B', 'C'] | "['A', 'B', 'C']" |
| roman_numeral | A child has received full Rabies vaccination in December 2018 and now presented with oozing wound on Great toe and the pet had vaccination also. Next line of management is | "{'I': 'No vaccine required', 'II': 'RIG + 5 doses of vaccine', 'III': '5 doses of vaccines only', 'IV': '2 doses of Rabies vaccine'}" | IV | "IV" |
| none_of_the_provided | A child has received full Rabies vaccination in December 2018 and now presented with oozing wound on Great toe and the pet had vaccination also. Next line of management is | "{'A': 'No vaccine required', 'B': 'RIG + 5 doses of vaccine', 'C': '5 doses of vaccines only', 'D': 'None of the provided options'}" | D | "D" |
| fixed_pos | A child has received full Rabies vaccination in December 2018 and now presented with oozing wound on Great toe and the pet had vaccination also. Next line of management is | "{'A': 'No vaccine required', 'B': 'RIG + 5 doses of vaccine', 'C': '5 doses of vaccines only', 'D': '2 doses of Rabies vaccine'}" | D | "D" |
| no_symbols | A child has received full Rabies vaccination in December 2018 and now presented with oozing wound on Great toe and the pet had vaccination also. Next line of management is | "{'A': 'No vaccine required', 'B': 'RIG + 5 doses of vaccine', 'C': '5 doses of vaccines only', 'D': '2 doses of Rabies vaccine'}" | 2 doses of Rabies vaccine | "2 doses of Rabies vaccine" |

#### id = `00b68a41-476c-4506-9961-35e03dd44243`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Which of the following enzyme is common between glycogenesis and glycogenolysis? | "{'A': 'Glycogen synthase', 'B': 'Glycogen phosphorylase', 'C': 'Glucan transferase', 'D': 'Phosphoglucomutase'}" | D | "D" |
| open | Which enzyme is common to both glycogenesis and glycogenolysis? | "{'A': 'Glycogen synthase', 'B': 'Glycogen phosphorylase', 'C': 'Glucan transferase', 'D': 'Phosphoglucomutase'}" | Phosphoglucomutase | "Phosphoglucomutase" |
| incorrect | Which of the following enzyme is common between glycogenesis and glycogenolysis? | "{'A': 'Glycogen synthase', 'B': 'Glycogen phosphorylase', 'C': 'Glucan transferase', 'D': 'Phosphoglucomutase'}" | ['A', 'B', 'C'] | "['A', 'B', 'C']" |
| roman_numeral | Which of the following enzyme is common between glycogenesis and glycogenolysis? | "{'I': 'Glycogen synthase', 'II': 'Glycogen phosphorylase', 'III': 'Glucan transferase', 'IV': 'Phosphoglucomutase'}" | IV | "IV" |
| none_of_the_provided | Which of the following enzyme is common between glycogenesis and glycogenolysis? | "{'A': 'Glycogen synthase', 'B': 'Glycogen phosphorylase', 'C': 'Glucan transferase', 'D': 'None of the provided options'}" | D | "D" |
| fixed_pos | Which of the following enzyme is common between glycogenesis and glycogenolysis? | "{'A': 'Glycogen synthase', 'B': 'Glycogen phosphorylase', 'C': 'Glucan transferase', 'D': 'Phosphoglucomutase'}" | D | "D" |
| no_symbols | Which of the following enzyme is common between glycogenesis and glycogenolysis? | "{'A': 'Glycogen synthase', 'B': 'Glycogen phosphorylase', 'C': 'Glucan transferase', 'D': 'Phosphoglucomutase'}" | Phosphoglucomutase | "Phosphoglucomutase" |

#### id = `00eb7f90-ec76-4b2a-89e8-9f716b070bfa`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Which of the following is the most common site for the occurrence of a basal cell carcinoma? | "{'A': 'Buccal mucosa', 'B': 'Hard palate', 'C': 'Skin of the lower lip', 'D': 'Dorsum of the tongue'}" | C | "C" |
| open | What is the most common site for the occurrence of a basal cell carcinoma? | "{'A': 'Buccal mucosa', 'B': 'Hard palate', 'C': 'Skin of the lower lip', 'D': 'Dorsum of the tongue'}" | Skin of the lower lip | "Skin of the lower lip" |
| incorrect | Which of the following is the most common site for the occurrence of a basal cell carcinoma? | "{'A': 'Buccal mucosa', 'B': 'Hard palate', 'C': 'Skin of the lower lip', 'D': 'Dorsum of the tongue'}" | ['A', 'B', 'D'] | "['A', 'B', 'D']" |
| roman_numeral | Which of the following is the most common site for the occurrence of a basal cell carcinoma? | "{'I': 'Buccal mucosa', 'II': 'Hard palate', 'III': 'Skin of the lower lip', 'IV': 'Dorsum of the tongue'}" | III | "III" |
| none_of_the_provided | Which of the following is the most common site for the occurrence of a basal cell carcinoma? | "{'A': 'Buccal mucosa', 'B': 'Hard palate', 'C': 'None of the provided options', 'D': 'Dorsum of the tongue'}" | C | "C" |
| fixed_pos | Which of the following is the most common site for the occurrence of a basal cell carcinoma? | "{'A': 'Buccal mucosa', 'B': 'Hard palate', 'C': 'Dorsum of the tongue', 'D': 'Skin of the lower lip'}" | D | "D" |
| no_symbols | Which of the following is the most common site for the occurrence of a basal cell carcinoma? | "{'A': 'Buccal mucosa', 'B': 'Hard palate', 'C': 'Skin of the lower lip', 'D': 'Dorsum of the tongue'}" | Skin of the lower lip | "Skin of the lower lip" |

### Source: `mmlu` — sample of 5 shared ids

#### id = `anatomy-000`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral | "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the fac... | A | "A" |
| open | What clinical deficit would result from compression of the facial nerve at the stylomastoid foramen on one side? | "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the fac... | paralysis of the facial muscles. | "paralysis of the facial muscles." |
| incorrect | A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral | "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the fac... | ['B', 'C', 'D'] | "['B', 'C', 'D']" |
| roman_numeral | A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral | "{'I': 'paralysis of the facial muscles.', 'II': 'paralysis of the facial muscles and loss of taste.', 'III': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'IV': 'paralysis of the... | I | "I" |
| none_of_the_provided | A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral | "{'A': 'None of the provided options', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial ... | A | "A" |
| fixed_pos | A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral | "{'A': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss o... | D | "D" |
| no_symbols | A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral | "{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the fac... | paralysis of the facial muscles. | "paralysis of the facial muscles." |

#### id = `anatomy-001`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | A "dished face" profile is often associated with | "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bon... | B | "B" |
| open | With which craniofacial abnormality is a "dished face" profile most often associated? | "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bon... | a recessive maxilla due to failure of elongation of the cranial base. | "a recessive maxilla due to failure of elongation of the cranial base." |
| incorrect | A "dished face" profile is often associated with | "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bon... | ['A', 'C', 'D'] | "['A', 'C', 'D']" |
| roman_numeral | A "dished face" profile is often associated with | "{'I': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'II': 'a recessive maxilla due to failure of elongation of the cranial base.', 'III': 'an enlarged frontal ... | II | "II" |
| none_of_the_provided | A "dished face" profile is often associated with | "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'None of the provided options', 'C': 'an enlarged frontal bone due to hydrocephaly.', 'D': 'defective ... | B | "B" |
| fixed_pos | A "dished face" profile is often associated with | "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'defective development of the maxillary air sinus.', 'C': 'an enlarged frontal bone due to hydrocephal... | D | "D" |
| no_symbols | A "dished face" profile is often associated with | "{'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.', 'B': 'a recessive maxilla due to failure of elongation of the cranial base.', 'C': 'an enlarged frontal bon... | a recessive maxilla due to failure of elongation of the cranial base. | "a recessive maxilla due to failure of elongation of the cranial base." |

#### id = `anatomy-002`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Which of the following best describes the structure that collects urine in the body? | "{'A': 'Bladder', 'B': 'Kidney', 'C': 'Ureter', 'D': 'Urethra'}" | A | "A" |
| open | What is the structure in the body that collects urine? | "{'A': 'Bladder', 'B': 'Kidney', 'C': 'Ureter', 'D': 'Urethra'}" | Bladder | "Bladder" |
| incorrect | Which of the following best describes the structure that collects urine in the body? | "{'A': 'Bladder', 'B': 'Kidney', 'C': 'Ureter', 'D': 'Urethra'}" | ['B', 'C', 'D'] | "['B', 'C', 'D']" |
| roman_numeral | Which of the following best describes the structure that collects urine in the body? | "{'I': 'Bladder', 'II': 'Kidney', 'III': 'Ureter', 'IV': 'Urethra'}" | I | "I" |
| none_of_the_provided | Which of the following best describes the structure that collects urine in the body? | "{'A': 'None of the provided options', 'B': 'Kidney', 'C': 'Ureter', 'D': 'Urethra'}" | A | "A" |
| fixed_pos | Which of the following best describes the structure that collects urine in the body? | "{'A': 'Urethra', 'B': 'Kidney', 'C': 'Ureter', 'D': 'Bladder'}" | D | "D" |
| no_symbols | Which of the following best describes the structure that collects urine in the body? | "{'A': 'Bladder', 'B': 'Kidney', 'C': 'Ureter', 'D': 'Urethra'}" | Bladder | "Bladder" |

#### id = `anatomy-003`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Which of the following structures is derived from ectomesenchyme? | "{'A': 'Motor neurons', 'B': 'Skeletal muscles', 'C': 'Melanocytes', 'D': 'Sweat glands'}" | C | "C" |
| open | Which structure in the human body is derived from ectomesenchyme? | "{'A': 'Motor neurons', 'B': 'Skeletal muscles', 'C': 'Melanocytes', 'D': 'Sweat glands'}" | Melanocytes | "Melanocytes" |
| incorrect | Which of the following structures is derived from ectomesenchyme? | "{'A': 'Motor neurons', 'B': 'Skeletal muscles', 'C': 'Melanocytes', 'D': 'Sweat glands'}" | ['A', 'B', 'D'] | "['A', 'B', 'D']" |
| roman_numeral | Which of the following structures is derived from ectomesenchyme? | "{'I': 'Motor neurons', 'II': 'Skeletal muscles', 'III': 'Melanocytes', 'IV': 'Sweat glands'}" | III | "III" |
| none_of_the_provided | Which of the following structures is derived from ectomesenchyme? | "{'A': 'Motor neurons', 'B': 'Skeletal muscles', 'C': 'None of the provided options', 'D': 'Sweat glands'}" | C | "C" |
| fixed_pos | Which of the following structures is derived from ectomesenchyme? | "{'A': 'Motor neurons', 'B': 'Skeletal muscles', 'C': 'Sweat glands', 'D': 'Melanocytes'}" | D | "D" |
| no_symbols | Which of the following structures is derived from ectomesenchyme? | "{'A': 'Motor neurons', 'B': 'Skeletal muscles', 'C': 'Melanocytes', 'D': 'Sweat glands'}" | Melanocytes | "Melanocytes" |

#### id = `anatomy-004`

| perturbation | question (truncated 200 chars) | options (parsed) | answer (raw) | answer (parsed) |
|---|---|---|---|---|
| mcq | Which of the following describes the cluster of blood capillaries found in each nephron in the kidney? | "{'A': 'Afferent arteriole', 'B': 'Glomerulus', 'C': 'Loop of Henle', 'D': 'Renal pelvis'}" | B | "B" |
| open | What is the name of the cluster of blood capillaries found in each nephron in the kidney? | "{'A': 'Afferent arteriole', 'B': 'Glomerulus', 'C': 'Loop of Henle', 'D': 'Renal pelvis'}" | Glomerulus | "Glomerulus" |
| incorrect | Which of the following describes the cluster of blood capillaries found in each nephron in the kidney? | "{'A': 'Afferent arteriole', 'B': 'Glomerulus', 'C': 'Loop of Henle', 'D': 'Renal pelvis'}" | ['A', 'C', 'D'] | "['A', 'C', 'D']" |
| roman_numeral | Which of the following describes the cluster of blood capillaries found in each nephron in the kidney? | "{'I': 'Afferent arteriole', 'II': 'Glomerulus', 'III': 'Loop of Henle', 'IV': 'Renal pelvis'}" | II | "II" |
| none_of_the_provided | Which of the following describes the cluster of blood capillaries found in each nephron in the kidney? | "{'A': 'Afferent arteriole', 'B': 'None of the provided options', 'C': 'Loop of Henle', 'D': 'Renal pelvis'}" | B | "B" |
| fixed_pos | Which of the following describes the cluster of blood capillaries found in each nephron in the kidney? | "{'A': 'Afferent arteriole', 'B': 'Renal pelvis', 'C': 'Loop of Henle', 'D': 'Glomerulus'}" | D | "D" |
| no_symbols | Which of the following describes the cluster of blood capillaries found in each nephron in the kidney? | "{'A': 'Afferent arteriole', 'B': 'Glomerulus', 'C': 'Loop of Henle', 'D': 'Renal pelvis'}" | Glomerulus | "Glomerulus" |

## 5. Index-Based vs ID-Based Join Stability

For one source family, compares how many rows a naive positional (index-based) join across two perturbation splits would agree with vs. how many an id-based join actually pairs correctly.

### Source: `medqa` (`medqa_mcq` vs `medqa_open`)
- n_rows: 1259 vs 1259 (equal)
- Positional (index-based) join: rows where id at position i is IDENTICAL between `medqa_mcq` and `medqa_open`: **1259 / 1259** (100.00%)
- ID-based join: ids present in BOTH splits (regardless of position): **1259 / 1259** (100.00%)
- CONCLUSION: row order is fully stable — a naive positional join would happen to produce the SAME result as an id-based join for this pair.

### Source: `medmcqa` (`medmcqa_mcq` vs `medmcqa_open`)
- n_rows: 1000 vs 1000 (equal)
- Positional (index-based) join: rows where id at position i is IDENTICAL between `medmcqa_mcq` and `medmcqa_open`: **1000 / 1000** (100.00%)
- ID-based join: ids present in BOTH splits (regardless of position): **1000 / 1000** (100.00%)
- CONCLUSION: row order is fully stable — a naive positional join would happen to produce the SAME result as an id-based join for this pair.

### Source: `mmlu` (`mmlu_mcq` vs `mmlu_open`)
- n_rows: 895 vs 895 (equal)
- Positional (index-based) join: rows where id at position i is IDENTICAL between `mmlu_mcq` and `mmlu_open`: **895 / 895** (100.00%)
- ID-based join: ids present in BOTH splits (regardless of position): **895 / 895** (100.00%)
- CONCLUSION: row order is fully stable — a naive positional join would happen to produce the SAME result as an id-based join for this pair.

## 6. Missing / Null Field Checks

| split | null_question | empty_question | null_options | empty_options | null_answer | empty_answer |
|---|---|---|---|---|---|---|
| medqa_mcq | 0 | 0 | 0 | 0 | 0 | 0 |
| medqa_open | 0 | 0 | 0 | 0 | 0 | 0 |
| medqa_incorrect | 0 | 0 | 0 | 0 | 0 | 0 |
| medqa_roman_numeral | 0 | 0 | 0 | 0 | 0 | 0 |
| medqa_none_of_the_provided | 0 | 0 | 0 | 0 | 0 | 0 |
| medqa_fixed_pos | 0 | 0 | 0 | 0 | 0 | 0 |
| medqa_no_symbols | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_mcq | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_open | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_incorrect | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_roman_numeral | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_none_of_the_provided | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_fixed_pos | 0 | 0 | 0 | 0 | 0 | 0 |
| medmcqa_no_symbols | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_mcq | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_open | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_incorrect | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_roman_numeral | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_none_of_the_provided | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_fixed_pos | 0 | 0 | 0 | 0 | 0 | 0 |
| mmlu_no_symbols | 0 | 0 | 0 | 0 | 0 | 0 |

## 7. Exact-Duplicate (Normalized) Questions Within Each Split

Normalization: lowercase, strip punctuation, collapse whitespace.

| split | n_rows | n_unique_normalized_questions | n_rows_involved_in_duplicates |
|---|---|---|---|
| medqa_mcq | 1259 | 1259 | 0 |
| medqa_open | 1259 | 1259 | 0 |
| medqa_incorrect | 1259 | 1259 | 0 |
| medqa_roman_numeral | 1259 | 1259 | 0 |
| medqa_none_of_the_provided | 1259 | 1259 | 0 |
| medqa_fixed_pos | 1259 | 1259 | 0 |
| medqa_no_symbols | 1259 | 1259 | 0 |
| medmcqa_mcq | 1000 | 998 | 4 |
| medmcqa_open | 1000 | 997 | 6 |
| medmcqa_incorrect | 1000 | 998 | 4 |
| medmcqa_roman_numeral | 1000 | 998 | 4 |
| medmcqa_none_of_the_provided | 1000 | 998 | 4 |
| medmcqa_fixed_pos | 1000 | 998 | 4 |
| medmcqa_no_symbols | 1000 | 998 | 4 |
| mmlu_mcq | 895 | 827 | 136 |
| mmlu_open | 895 | 838 | 114 |
| mmlu_incorrect | 895 | 827 | 136 |
| mmlu_roman_numeral | 895 | 827 | 136 |
| mmlu_none_of_the_provided | 895 | 827 | 136 |
| mmlu_fixed_pos | 895 | 827 | 136 |
| mmlu_no_symbols | 895 | 827 | 136 |

## 8. What Each Perturbation Type Actually Does (Direct Inspection)

Method: for each source, take one id shared across all 7 perturbations, and diff the `mcq` version's `options`/`answer` against each other perturbation's `options`/`answer` for that exact same id.

### Source: `medqa`, inspected id = `0000`

- **mcq (baseline)**: options=`{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`, answer=`B` (raw: `B`)
- **open**: options=`{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`, answer=`Tell the attending that he cannot fail to disclose this mistake` (raw: `Tell the attending that he cannot fail to disclose this mistake`)
- **incorrect**: options=`{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`, answer=`['A', 'C', 'D']` (raw: `['A', 'C', 'D']`)
- **roman_numeral**: options=`{'I': 'Disclose the error to the patient and put it in the operative report', 'II': 'Tell the attending that he cannot fail to disclose this mistake', 'III': 'Report the physician to the ethics committee', 'IV': 'Refuse to dictate the operative report'}`, answer=`II` (raw: `II`)
- **none_of_the_provided**: options=`{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'None of the provided options', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`, answer=`B` (raw: `B`)
- **fixed_pos**: options=`{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Refuse to dictate the operative report', 'C': 'Report the physician to the ethics committee', 'D': 'Tell the attending that he cannot fail to disclose this mistake'}`, answer=`D` (raw: `D`)
- **no_symbols**: options=`{'A': 'Disclose the error to the patient and put it in the operative report', 'B': 'Tell the attending that he cannot fail to disclose this mistake', 'C': 'Report the physician to the ethics committee', 'D': 'Refuse to dictate the operative report'}`, answer=`Tell the attending that he cannot fail to disclose this mistake` (raw: `Tell the attending that he cannot fail to disclose this mistake`)

### Source: `medmcqa`, inspected id = `0009b2fd-7e72-4ed0-b486-92ccb24e43f3`

- **mcq (baseline)**: options=`{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}`, answer=`A` (raw: `A`)
- **open**: options=`{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}`, answer=`IgG Anti HbcAg` (raw: `IgG Anti HbcAg`)
- **incorrect**: options=`{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}`, answer=`['B', 'C', 'D']` (raw: `['B', 'C', 'D']`)
- **roman_numeral**: options=`{'I': 'IgG Anti HbcAg', 'II': 'HBsAg', 'III': 'IgG Anti HBsAG', 'IV': 'Anti Hbs'}`, answer=`I` (raw: `I`)
- **none_of_the_provided**: options=`{'A': 'None of the provided options', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}`, answer=`A` (raw: `A`)
- **fixed_pos**: options=`{'A': 'Anti Hbs', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'IgG Anti HbcAg'}`, answer=`D` (raw: `D`)
- **no_symbols**: options=`{'A': 'IgG Anti HbcAg', 'B': 'HBsAg', 'C': 'IgG Anti HBsAG', 'D': 'Anti Hbs'}`, answer=`IgG Anti HbcAg` (raw: `IgG Anti HbcAg`)

### Source: `mmlu`, inspected id = `anatomy-000`

- **mcq (baseline)**: options=`{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`, answer=`A` (raw: `A`)
- **open**: options=`{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`, answer=`paralysis of the facial muscles.` (raw: `paralysis of the facial muscles.`)
- **incorrect**: options=`{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`, answer=`['B', 'C', 'D']` (raw: `['B', 'C', 'D']`)
- **roman_numeral**: options=`{'I': 'paralysis of the facial muscles.', 'II': 'paralysis of the facial muscles and loss of taste.', 'III': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'IV': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`, answer=`I` (raw: `I`)
- **none_of_the_provided**: options=`{'A': 'None of the provided options', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`, answer=`A` (raw: `A`)
- **fixed_pos**: options=`{'A': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles.'}`, answer=`D` (raw: `D`)
- **no_symbols**: options=`{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}`, answer=`paralysis of the facial muscles.` (raw: `paralysis of the facial muscles.`)

### Interpretation notes (grounded in the printed diffs above — verify against actual output before trusting)

- These are observations from ONE sampled id per source; see Section 4 for additional samples. Treat perturbation-level generalizations as provisional unless consistent across all inspected samples.

## Summary

See sections above for exact, run-grounded numbers. No values in this file were hand-written/fabricated; all were produced by executing this script against the on-disk dataset at the path listed at the top.
