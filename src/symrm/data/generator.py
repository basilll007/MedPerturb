"""Counterfactual clinical vignette dataset generator for SymRM.

Generates deterministic triplets (base, edited, near-miss) across 10 clinical rules:
- Base: normal clinical presentation, default action correct.
- Edited: exactly ONE decisive clinical fact changed, rule action correct.
- Near-miss: concept mentioned without triggering the rule, default action correct.

Critical Quality and Clinical Guidelines (Gate 2a):
1. Raw values only in identical field format across base, edited, and near-miss:
   - Renal: "Renal function: eGFR <value> mL/min/1.73m2, serum creatinine <value> mg/dL."
     or "Renal function: creatinine clearance <value> mL/min, serum creatinine <value> mg/dL."
     Base items MUST include eGFR/CrCl values in normal range (85-110).
   - Pregnancy: "Reproductive status: urine beta-hCG <positive|negative>." (gestational age only when positive).
     Base items use "Reproductive status: non-pregnant."
   - Allergy: "Drug allergies: <drug> (<reaction>)" or "Drug allergies: none."
     Separate "Family history:" field in all 3 items ("noncontributory" when not near-miss).
2. Zero banned tokens: severe, marked, mild, moderate, impairment, dysfunction,
   insufficiency, adequate, preserved, safely, tolerated, confirming, reference range, band,
   threshold annotations (>= 60, < 30), or parenthetical interpretations.
3. Base, edited, and near-miss prompts within a scenario differ ONLY in the decisive field
   and (for near-miss) family-history field.
4. Age spans 25-80 per rule; sex is mixed 50/50 where clinically valid (female-only for pregnancy & primary dysmenorrhea).
5. All items tagged "data_version": "v0-unreviewed".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel


class StructuredFacts(BaseModel):
    rule_id: str
    indication: str
    age: int
    sex: str
    egfr: Optional[float] = None
    crcl: Optional[float] = None
    pregnant: bool = False
    beta_hcg: Optional[str] = None
    pregnancy_planning: bool = False
    allergy: Optional[str] = None
    allergy_reaction: Optional[str] = None
    family_allergy: Optional[str] = None
    unrelated_allergy: Optional[str] = None


class ClinicalItem(BaseModel):
    item_id: str
    template_id: str
    rule_id: str
    family: str
    split: str
    item_type: str  # 'base', 'edited', 'near_miss', 'null_edit'
    indication: str
    prompt: str
    chosen: str
    rejected: str
    default_option: str
    alternative_option: str
    chosen_is_default: bool
    decisive_fact_present: bool
    decisive_fact_type: str
    structured_facts: Dict[str, Any]
    data_version: str = "v0-unreviewed"


class SymbolicVerifier:
    """Deterministic rule verifier over structured clinical facts."""

    @staticmethod
    def verify(facts: StructuredFacts, rule_spec: Dict[str, Any]) -> Tuple[bool, str]:
        """Returns (rule_fired, correct_action)."""
        rule_id = rule_spec["rule_id"]
        default_act = rule_spec["default_action"]
        rule_act = rule_spec["rule_action"]

        rule_fired = False

        if rule_id == "renal_01":
            if facts.egfr is not None and facts.egfr < 30.0:
                rule_fired = True
        elif rule_id == "renal_02":
            if facts.crcl is not None and facts.crcl < 30.0:
                rule_fired = True
        elif rule_id == "renal_03":
            if facts.crcl is not None and facts.crcl < 30.0:
                rule_fired = True
        elif rule_id == "renal_04":
            if facts.crcl is not None and 15.0 <= facts.crcl < 30.0:
                rule_fired = True
        elif rule_id in ["preg_01", "preg_02", "preg_03"]:
            if facts.pregnant is True:
                rule_fired = True
        elif rule_id == "allergy_01":
            if facts.allergy == "penicillin" and facts.allergy_reaction == "anaphylaxis":
                rule_fired = True
        elif rule_id == "allergy_02":
            if facts.allergy == "sulfonamide" and facts.allergy_reaction in ["anaphylaxis", "stevens_johnson"]:
                rule_fired = True
        elif rule_id == "allergy_03":
            if facts.allergy == "aspirin" and facts.allergy_reaction == "bronchospasm_polyps":
                rule_fired = True

        correct_action = rule_act if rule_fired else default_act
        return rule_fired, correct_action


class DatasetGenerator:
    """Generates grouped counterfactual vignette datasets according to rules.yaml."""

    def __init__(self, rules_path: str = "configs/symrm/rules.yaml", seed: int = 42):
        self.rules_path = Path(rules_path)
        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.rules_data = yaml.safe_load(f)
        self.rules_list = self.rules_data["rules"]
        self.rules_by_id = {r["rule_id"]: r for r in self.rules_list}
        self.split_defs = self.rules_data["split_definitions"]
        self.seed = seed

    def _get_rule_split(self, rule_id: str) -> str:
        for split_name, spec in self.split_defs.items():
            if rule_id in spec.get("rule_ids", []):
                return split_name
        return "train"

    @staticmethod
    def get_article_for_age(age: int) -> str:
        """Determines 'A' vs 'An' for age."""
        if str(age).startswith("8"):
            return "An"
        return "A"

    def generate_all(self, templates_per_rule: int = 30) -> List[ClinicalItem]:
        """Generates templates_per_rule triplets for each of the 10 rules (total items = 30 * 10 * 3 = 900)."""
        all_items: List[ClinicalItem] = []

        for rule in self.rules_list:
            rule_items = self._generate_rule_triplets(rule, count=templates_per_rule)
            all_items.extend(rule_items)

        return all_items

    def _generate_rule_triplets(self, rule: Dict[str, Any], count: int) -> List[ClinicalItem]:
        rule_id = rule["rule_id"]
        family = rule["family"]
        split = self._get_rule_split(rule_id)
        indication = rule["indication"]
        default_act = rule["default_action"]
        rule_act = rule["rule_action"]

        items: List[ClinicalItem] = []

        settings = [
            "in the primary care clinic",
            "at the outpatient ambulatory care center",
            "during a follow-up consultation in clinic",
            "at the urgent care center",
            "in the medical clinic following diagnostic workup",
            "in the general outpatient medical clinic",
            "at the community health center clinic"
        ]

        distractor_labs = [
            "Serum potassium 4.2 mEq/L, sodium 140 mEq/L, AST 22 U/L, ALT 24 U/L.",
            "Complete metabolic panel shows normal hepatic transaminases and normal serum electrolytes.",
            "Vital signs: pulse 74 bpm, respirations 16/min, afebrile.",
            "Laboratory studies show normal liver function tests and serum electrolytes.",
            "Physical examination shows normal heart sounds, clear breath sounds, and no peripheral edema.",
            "Routine chemistry profile demonstrates normal serum sodium, potassium, and chloride levels.",
            "Cardiovascular and pulmonary examinations are unremarkable with clear lung fields bilaterally.",
            "Basic diagnostic panel shows normal serum bilirubin, alkaline phosphatase, and normal bicarbonate.",
            "Resting vital signs demonstrate blood pressure 124/78 mmHg, heart rate 72 bpm, room air saturation 99%.",
            "Comprehensive metabolic screening confirms normal plasma electrolytes and normal hepatic enzymes.",
            "General physical examination reveals no acute distress, warm well-perfused extremities, and clear breath sounds."
        ]

        for i in range(count):
            template_id = f"tpl_{rule_id}_{i+1:03d}"

            # Age strictly spans 25 to 80
            age = 25 + int(round(i * (80 - 25) / (count - 1)))
            article = self.get_article_for_age(age)

            # Sex distribution:
            # - pregnancy rules (preg_01..03) & dysmenorrhea (allergy_03) MUST be female only.
            # - all other rules are mixed 50/50 male/female.
            if family == "pregnancy" or rule_id == "allergy_03":
                sex = "female"
            else:
                sex = "female" if (i % 2 == 0) else "male"

            setting = settings[i % len(settings)]
            distractor = distractor_labs[i % len(distractor_labs)]
            presentation = self._get_indication_presentation(rule_id, i)

            # Common baseline values
            base_cr = round(0.85 + (i % 5) * 0.04, 2)
            base_egfr = 88.0 + (i % 23)
            base_crcl = 90.0 + (i % 21)

            # -----------------------------------------------------------------
            # 1. BASE ITEM
            # -----------------------------------------------------------------
            base_facts = StructuredFacts(
                rule_id=rule_id,
                indication=indication,
                age=age,
                sex=sex,
            )

            if rule_id == "renal_01":
                base_facts.egfr = base_egfr
                base_facts.crcl = base_egfr + 2
                renal_text = f"Renal function: eGFR {base_egfr:.0f} mL/min/1.73m2, serum creatinine {base_cr} mg/dL."
            elif family == "renal":
                base_facts.crcl = base_crcl
                base_facts.egfr = base_crcl - 2
                renal_text = f"Renal function: creatinine clearance {base_crcl:.0f} mL/min, serum creatinine {base_cr} mg/dL."
            else:
                base_facts.egfr = base_egfr
                renal_text = f"Renal function: eGFR {base_egfr:.0f} mL/min/1.73m2, serum creatinine {base_cr} mg/dL."

            if sex == "female":
                repro_text = "Reproductive status: non-pregnant."
                base_facts.pregnant = False
                base_facts.beta_hcg = "negative"
            else:
                repro_text = "Reproductive status: not applicable."

            allergy_text = "Drug allergies: none."
            family_text = "Family history: noncontributory."

            base_prompt = self._compose_prompt(
                article=article, age=age, sex=sex, setting=setting, indication=indication,
                presentation=presentation, renal_text=renal_text, repro_text=repro_text,
                allergy_text=allergy_text, family_text=family_text, distractor_text=distractor
            )

            # Verify base
            fired, correct = SymbolicVerifier.verify(base_facts, rule)
            assert not fired, f"Base item triggered rule {rule_id}"
            assert correct == default_act

            items.append(ClinicalItem(
                item_id=f"{template_id}_base",
                template_id=template_id,
                rule_id=rule_id,
                family=family,
                split=split,
                item_type="base",
                indication=indication,
                prompt=base_prompt,
                chosen=default_act,
                rejected=rule_act,
                default_option=default_act,
                alternative_option=rule_act,
                chosen_is_default=True,
                decisive_fact_present=False,
                decisive_fact_type="none",
                structured_facts=base_facts.model_dump(),
            ))

            # -----------------------------------------------------------------
            # 2. EDITED ITEM (Counterfactual Trigger)
            # -----------------------------------------------------------------
            edited_facts = StructuredFacts(
                rule_id=rule_id,
                indication=indication,
                age=age,
                sex=sex,
            )

            renal_text_ed = renal_text
            repro_text_ed = repro_text
            allergy_text_ed = allergy_text
            family_text_ed = family_text
            decisive_type = "none"

            if rule_id == "renal_01":
                egfr_val = 16.0 + (i % 13)  # 16-28 mL/min (< 30)
                cr_ed = round(3.2 + (i % 6) * 0.1, 2)
                edited_facts.egfr = egfr_val
                edited_facts.crcl = egfr_val + 2
                renal_text_ed = f"Renal function: eGFR {egfr_val:.0f} mL/min/1.73m2, serum creatinine {cr_ed} mg/dL."
                decisive_type = "renal_impairment"
            elif rule_id == "renal_02":
                crcl_val = 16.0 + (i % 13)  # 16-28 mL/min (< 30)
                cr_ed = round(2.8 + (i % 6) * 0.1, 2)
                edited_facts.crcl = crcl_val
                edited_facts.egfr = crcl_val - 2
                renal_text_ed = f"Renal function: creatinine clearance {crcl_val:.0f} mL/min, serum creatinine {cr_ed} mg/dL."
                decisive_type = "renal_impairment"
            elif rule_id == "renal_03":
                crcl_val = 16.0 + (i % 13)  # 16-28 mL/min (< 30)
                cr_ed = round(2.6 + (i % 6) * 0.1, 2)
                edited_facts.crcl = crcl_val
                edited_facts.egfr = crcl_val
                renal_text_ed = f"Renal function: creatinine clearance {crcl_val:.0f} mL/min, serum creatinine {cr_ed} mg/dL."
                decisive_type = "renal_impairment"
            elif rule_id == "renal_04":
                crcl_val = 16.0 + (i % 13)  # 16-28 mL/min (15-29)
                cr_ed = round(2.5 + (i % 6) * 0.1, 2)
                edited_facts.crcl = crcl_val
                edited_facts.egfr = crcl_val
                renal_text_ed = f"Renal function: creatinine clearance {crcl_val:.0f} mL/min, serum creatinine {cr_ed} mg/dL."
                decisive_type = "renal_impairment"
            elif family == "pregnancy":
                edited_facts.pregnant = True
                edited_facts.beta_hcg = "positive"
                gest_weeks = 6 + (i % 9)  # 6-14 weeks
                repro_text_ed = f"Reproductive status: urine beta-hCG positive ({gest_weeks} weeks gestation)."
                decisive_type = "pregnancy"
            elif rule_id == "allergy_01":
                edited_facts.allergy = "penicillin"
                edited_facts.allergy_reaction = "anaphylaxis"
                allergy_text_ed = "Drug allergies: penicillin (anaphylaxis)."
                decisive_type = "drug_allergy"
            elif rule_id == "allergy_02":
                edited_facts.allergy = "sulfonamide"
                rxn = "anaphylaxis" if (i % 2 == 0) else "Stevens-Johnson syndrome"
                edited_facts.allergy_reaction = "anaphylaxis" if (i % 2 == 0) else "stevens_johnson"
                allergy_text_ed = f"Drug allergies: sulfonamide ({rxn})."
                decisive_type = "drug_allergy"
            elif rule_id == "allergy_03":
                edited_facts.allergy = "aspirin"
                edited_facts.allergy_reaction = "bronchospasm_polyps"
                allergy_text_ed = "Drug allergies: aspirin (bronchospasm and nasal polyps)."
                decisive_type = "drug_allergy"

            edited_prompt = self._compose_prompt(
                article=article, age=age, sex=sex, setting=setting, indication=indication,
                presentation=presentation, renal_text=renal_text_ed, repro_text=repro_text_ed,
                allergy_text=allergy_text_ed, family_text=family_text_ed, distractor_text=distractor
            )

            # Verify edited
            fired, correct = SymbolicVerifier.verify(edited_facts, rule)
            assert fired, f"Edited item failed to trigger rule {rule_id}"
            assert correct == rule_act

            items.append(ClinicalItem(
                item_id=f"{template_id}_edited",
                template_id=template_id,
                rule_id=rule_id,
                family=family,
                split=split,
                item_type="edited",
                indication=indication,
                prompt=edited_prompt,
                chosen=rule_act,
                rejected=default_act,
                default_option=default_act,
                alternative_option=rule_act,
                chosen_is_default=False,
                decisive_fact_present=True,
                decisive_fact_type=decisive_type,
                structured_facts=edited_facts.model_dump(),
            ))

            # -----------------------------------------------------------------
            # 3. NEAR-MISS ITEM (Concept Mentioned Without Triggering Rule)
            # -----------------------------------------------------------------
            nm_facts = StructuredFacts(
                rule_id=rule_id,
                indication=indication,
                age=age,
                sex=sex,
            )

            renal_text_nm = renal_text
            repro_text_nm = repro_text
            allergy_text_nm = allergy_text
            family_text_nm = family_text

            if rule_id == "renal_01":
                egfr_val = 46.0 + (i % 13)  # 46-58 mL/min (45-59)
                cr_nm = round(1.3 + (i % 5) * 0.05, 2)
                nm_facts.egfr = egfr_val
                nm_facts.crcl = egfr_val + 2
                renal_text_nm = f"Renal function: eGFR {egfr_val:.0f} mL/min/1.73m2, serum creatinine {cr_nm} mg/dL."
            elif rule_id == "renal_02":
                crcl_val = 42.0 + (i % 17)  # 42-58 mL/min (40-59)
                cr_nm = round(1.4 + (i % 5) * 0.05, 2)
                nm_facts.crcl = crcl_val
                nm_facts.egfr = crcl_val - 2
                renal_text_nm = f"Renal function: creatinine clearance {crcl_val:.0f} mL/min, serum creatinine {cr_nm} mg/dL."
            elif rule_id == "renal_03":
                crcl_val = 65.0 + (i % 18)  # 65-82 mL/min (>= 60)
                cr_nm = round(1.0 + (i % 4) * 0.05, 2)
                nm_facts.crcl = crcl_val
                nm_facts.egfr = crcl_val
                renal_text_nm = f"Renal function: creatinine clearance {crcl_val:.0f} mL/min, serum creatinine {cr_nm} mg/dL."
            elif rule_id == "renal_04":
                crcl_val = 66.0 + (i % 18)  # 66-83 mL/min (>= 60)
                cr_nm = round(0.9 + (i % 4) * 0.05, 2)
                nm_facts.crcl = crcl_val
                nm_facts.egfr = crcl_val
                renal_text_nm = f"Renal function: creatinine clearance {crcl_val:.0f} mL/min, serum creatinine {cr_nm} mg/dL."
            elif family == "pregnancy":
                # Near-miss is negative hCG only
                nm_facts.pregnant = False
                nm_facts.beta_hcg = "negative"
                repro_text_nm = "Reproductive status: urine beta-hCG negative."
            elif rule_id == "allergy_01":
                nm_facts.allergy = "none"
                relative = ["mother", "father", "sibling"][i % 3]
                nm_facts.family_allergy = f"penicillin_anaphylaxis_{relative}"
                family_text_nm = f"Family history: penicillin allergy (anaphylaxis) in {relative}."
            elif rule_id == "allergy_02":
                nm_facts.allergy = "none"
                relative = ["sibling", "mother", "father"][i % 3]
                nm_facts.family_allergy = f"sulfonamide_sjs_{relative}"
                family_text_nm = f"Family history: sulfonamide allergy (Stevens-Johnson syndrome) in {relative}."
            elif rule_id == "allergy_03":
                nm_facts.allergy = "none"
                relative = ["father", "mother", "sibling"][i % 3]
                nm_facts.family_allergy = f"aspirin_aerd_{relative}"
                family_text_nm = f"Family history: aspirin sensitivity (bronchospasm and nasal polyps) in {relative}."

            nm_prompt = self._compose_prompt(
                article=article, age=age, sex=sex, setting=setting, indication=indication,
                presentation=presentation, renal_text=renal_text_nm, repro_text=repro_text_nm,
                allergy_text=allergy_text_nm, family_text=family_text_nm, distractor_text=distractor
            )

            # Verify near-miss
            fired, correct = SymbolicVerifier.verify(nm_facts, rule)
            assert not fired, f"Near-miss item triggered rule {rule_id}"
            assert correct == default_act

            items.append(ClinicalItem(
                item_id=f"{template_id}_near_miss",
                template_id=template_id,
                rule_id=rule_id,
                family=family,
                split=split,
                item_type="near_miss",
                indication=indication,
                prompt=nm_prompt,
                chosen=default_act,
                rejected=rule_act,
                default_option=default_act,
                alternative_option=rule_act,
                chosen_is_default=True,
                decisive_fact_present=False,
                decisive_fact_type="none",
                structured_facts=nm_facts.model_dump(),
            ))

            # -----------------------------------------------------------------
            # 4. NULL-EDIT ITEM (Paraphrasing Non-Decisive Fields)
            # -----------------------------------------------------------------
            null_setting = settings[(i + 1) % len(settings)]
            null_distractor = distractor_labs[(i + 1) % len(distractor_labs)]

            null_prompt = self._compose_prompt(
                article=article, age=age, sex=sex, setting=null_setting, indication=indication,
                presentation=presentation, renal_text=renal_text, repro_text=repro_text,
                allergy_text=allergy_text, family_text=family_text, distractor_text=null_distractor
            )

            items.append(ClinicalItem(
                item_id=f"{template_id}_null_edit",
                template_id=template_id,
                rule_id=rule_id,
                family=family,
                split=split,
                item_type="null_edit",
                indication=indication,
                prompt=null_prompt,
                chosen=default_act,
                rejected=rule_act,
                default_option=default_act,
                alternative_option=rule_act,
                chosen_is_default=True,
                decisive_fact_present=False,
                decisive_fact_type="none",
                structured_facts=base_facts.model_dump(),
            ))

        return items

    def _compose_prompt(
        self,
        article: str,
        age: int,
        sex: str,
        setting: str,
        indication: str,
        presentation: str,
        renal_text: str,
        repro_text: str,
        allergy_text: str,
        family_text: str,
        distractor_text: str,
    ) -> str:
        prompt = (
            f"Clinical Vignette:\n"
            f"{article} {age}-year-old {sex} presents {setting} for management of {indication}. {presentation}\n"
            f"{renal_text}\n"
            f"{repro_text}\n"
            f"{allergy_text}\n"
            f"{family_text}\n"
            f"{distractor_text}\n\n"
            f"Question: What is the most appropriate pharmacological prescription order for this patient?"
        )
        return prompt

    def _get_indication_presentation(self, rule_id: str, index: int) -> str:
        """Generates varied clinical indication presentations with zero banned tokens."""
        presentations = {
            "renal_01": [
                "Patient presents with suboptimal glycemic control on diet and exercise alone, with fasting glucose 174 mg/dL and HbA1c 8.4%.",
                "Routine metabolic follow-up demonstrates persistent hyperglycemia, fasting blood glucose 188 mg/dL, and HbA1c 8.7%.",
                "Patient notes polyuria and polydipsia; point-of-care capillary glucose is 196 mg/dL with recent HbA1c of 8.6%.",
                "Evaluation for outpatient glycemic optimization shows fasting venous glucose 182 mg/dL and HbA1c 8.5%.",
                "Follow-up laboratory assessment reveals uncontrolled blood sugar with HbA1c 8.8% and morning fingerstick glucose 179 mg/dL.",
                "Metabolic re-evaluation reveals elevated glycated hemoglobin at 8.3% and random blood glucose 192 mg/dL.",
            ],
            "renal_02": [
                "Diagnostic lower extremity venous duplex ultrasound demonstrates acute occlusive deep vein thrombosis involving the left popliteal vein.",
                "Patient presents with acute unilateral right calf swelling and tenderness; duplex ultrasound demonstrates non-compressible common femoral DVT.",
                "Computed tomography pulmonary angiogram demonstrates acute segmental pulmonary embolism without hemodynamic instability.",
                "Venous compression ultrasonography demonstrates acute non-occlusive thrombus in the right femoral vein.",
                "Patient reports 3 days of progressive left leg aching and swelling; duplex scan demonstrates acute left calf deep venous thrombosis.",
                "Chest CT with contrast reveals acute subsegmental pulmonary thromboembolism in the right lower lobe.",
            ],
            "renal_03": [
                "Patient presents with acute dysuria, urinary frequency, and suprapubic discomfort for 48 hours without fever or flank pain.",
                "Urinalysis reveals positive leukocyte esterase, positive nitrites, and 25 WBC/HPF consistent with acute uncomplicated cystitis.",
                "Symptom onset 2 days ago with painful urination, increased urinary urgency, and absence of systemic signs or costovertebral tenderness.",
                "Patient describes acute onset burning with micturition and persistent pelvic pressure without back pain.",
                "Urine dipstick demonstrates positive leukocyte esterase and bacteria, accompanying 3 days of painful frequent urination.",
                "Outpatient presentation for painful burning micturition, increased bladder fullness, and absence of constitutional symptoms.",
            ],
            "renal_04": [
                "Patient presents with sharp burning neuropathic pain and hyperalgesia along the T5-T6 dermatome following herpes zoster rash resolution.",
                "Neuropathic pain assessment reveals sharp lancinating dysesthesias in a postherpetic distribution scoring 7/10 on the pain scale.",
                "Persistent dermatomal allodynia and postherpetic neuralgic pain for 3 months refractory to over-the-counter analgesics.",
                "Clinical review demonstrates chronic burning dysesthetic pain along the thoracic dermatome after shingles healing.",
                "Patient reports persistent shooting and tingling cutaneous sensations across the left flank following prior shingles.",
                "Localized postherpetic neuropathic pain in the right T8 dermatomal area causing sleep disruption and touch sensitivity.",
            ],
            "preg_01": [
                "Clinic blood pressure measurements show persistent elevation averaging 148/94 mmHg on three separate seated determinations.",
                "Patient presents for routine cardiovascular management with repeated clinic blood pressure readings of 152/96 mmHg.",
                "Ambulatory blood pressure monitoring demonstrates sustained stage 1 essential hypertension with daytime systolic readings averaging 146 mmHg.",
                "Serial clinic evaluations document persistent asymptomatic arterial hypertension with blood pressure 150/92 mmHg.",
                "Cardiovascular follow-up demonstrates sustained elevated blood pressure at 146/94 mmHg without end-organ target damage.",
                "Office blood pressure screening demonstrates elevated seated measurements of 154/95 mmHg on separate visits.",
            ],
            "preg_02": [
                "Patient presents with active inflammatory symmetric polyarthritis involving bilateral MCP and PIP joints with morning stiffness lasting 90 minutes.",
                "Rheumatologic evaluation reveals active joint synovitis, elevated ESR 42 mm/hr, and positive rheumatoid factor.",
                "Persistent bilateral wrist and hand joint pain with palpable synovitis requiring initiation of disease-modifying antirheumatic therapy.",
                "Patient exhibits symmetric joint tenderness in the metacarpophalangeal joints and prolonged morning stiffness.",
                "Musculoskeletal exam reveals boggy synovial thickening of the small joints of both hands and wrists.",
                "Active polyarticular inflammatory disease with morning stiffness and bilateral wrist involvement.",
            ],
            "preg_03": [
                "Physical examination reveals an expanding 8-cm erythematous annular lesion with central clearing consistent with erythema migrans following tick exposure.",
                "Patient noticed an expanding bullseye-pattern annular rash on the right thigh 10 days after hiking in an endemic wooded area.",
                "Solitary erythema migrans skin lesion measuring 7 cm in diameter accompanied by fatigue and low-grade myalgias.",
                "Dermatologic assessment demonstrates single expanding targetoid circular erythematous patch on the lower extremity.",
                "Patient presents with 6-cm expanding circular erythematous skin lesion following deer tick bite in an endemic region.",
                "Characteristic expanding annular cutaneous plaque with central clearing on the trunk following outdoor exposure.",
            ],
            "allergy_01": [
                "Patient presents with acute sore throat, odynophagia, fever of 38.5 C, tonsillar exudates, and positive rapid streptococcal antigen test.",
                "Physical exam demonstrates pharyngeal erythema, bilateral palatine tonsillar enlargement with purulent exudate, and tender anterior cervical nodes.",
                "Rapid Group A Strep test is positive in the setting of abrupt onset pharyngitis, absence of cough, and temp 38.4 C.",
                "Patient presents with fever, painful swallowing, bilateral tonsillopharyngeal exudates, and positive rapid antigen detection.",
                "Examination shows beefy red posterior pharynx, palatal petechiae, anterior cervical lymphadenitis, and positive Strep test.",
                "Acute febrile pharyngitis with bilateral tonsillar exudates and positive point-of-care rapid streptococcal test.",
            ],
            "allergy_02": [
                "Patient presents with acute onset of burning on urination, frequency, and bladder fullness consistent with uncomplicated acute cystitis.",
                "Urinalysis demonstrates positive nitrites, 3+ leukocyte esterase, and bacteriuria; physical exam shows no costovertebral angle tenderness.",
                "Acute dysuria and increased daytime urinary frequency for 36 hours; point-of-care dipstick positive for nitrites and leukocyte esterase.",
                "Patient presents with urinary urgency, suprapubic discomfort, and painful urination without fever or flank tenderness.",
                "Urine analysis reveals positive leukocyte esterase and nitrites in the setting of 2 days of dysuria and frequency.",
                "Outpatient evaluation reveals painful urination and increased urinary frequency without systemic signs.",
            ],
            "allergy_03": [
                "Patient presents with recurrent crampy lower abdominal and pelvic pain beginning on the first day of menses, associated with nausea.",
                "Patient presents for menstrual pain management with cyclic midline pelvic cramping that peaks on days 1 and 2 of menstruation.",
                "Patient reports monthly spasmodic lower pelvic pain coinciding with menses onset, without dyspareunia or abnormal uterine bleeding.",
                "Recurrent suprapubic cramping pain that begins several hours prior to menstrual flow and radiates to the thighs.",
                "Patient describes cyclic midline menstrual cramps during the first 48 hours of menses without pelvic pathology.",
                "Painful cyclic lower abdominal cramping accompanying the onset of menses every month.",
            ],
        }
        options = presentations.get(rule_id, ["Clinical presentation consistent with stated indication."])
        return options[index % len(options)]

    def save_datasets(self, items: List[ClinicalItem], output_dir: str = "results/symrm/data") -> Dict[str, str]:
        """Saves generated items split into train, near_ood, far_ood, and combined JSONL files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train_items = [it.model_dump() for it in items if it.split == "train"]
        near_ood_items = [it.model_dump() for it in items if it.split == "near_ood"]
        far_ood_items = [it.model_dump() for it in items if it.split == "far_ood"]
        all_items_dump = [it.model_dump() for it in items]

        paths = {
            "train": str(out_path / "train.jsonl"),
            "near_ood": str(out_path / "near_ood.jsonl"),
            "far_ood": str(out_path / "far_ood.jsonl"),
            "all": str(out_path / "all_items.jsonl"),
            "metadata": str(out_path / "metadata.json"),
        }

        with open(paths["train"], "w", encoding="utf-8") as f:
            for item in train_items:
                f.write(json.dumps(item) + "\n")

        with open(paths["near_ood"], "w", encoding="utf-8") as f:
            for item in near_ood_items:
                f.write(json.dumps(item) + "\n")

        with open(paths["far_ood"], "w", encoding="utf-8") as f:
            for item in far_ood_items:
                f.write(json.dumps(item) + "\n")

        with open(paths["all"], "w", encoding="utf-8") as f:
            for item in all_items_dump:
                f.write(json.dumps(item) + "\n")

        metadata = {
            "data_version": "v0-unreviewed",
            "total_items": len(items),
            "num_templates": len(items) // 4,
            "train_items": len(train_items),
            "near_ood_items": len(near_ood_items),
            "far_ood_items": len(far_ood_items),
            "rules_count": len(self.rules_list),
            "seed": self.seed,
        }
        with open(paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return paths


def generate_dataset(rules_path: str = "configs/symrm/rules.yaml", output_dir: str = "results/symrm/data") -> List[ClinicalItem]:
    gen = DatasetGenerator(rules_path=rules_path)
    items = gen.generate_all(templates_per_rule=30)
    gen.save_datasets(items, output_dir=output_dir)
    return items


if __name__ == "__main__":
    items = generate_dataset()
    print(f"Generated {len(items)} items across {len(items)//4} templates.")
