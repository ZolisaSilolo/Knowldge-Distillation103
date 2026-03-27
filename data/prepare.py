"""
prepare.py — Download, preprocess, and generate clinical training data.

This script:
1. Downloads open medical QA datasets (MedQA subset, HealthCareMagic)
2. Filters for primary-care-relevant questions
3. Generates synthetic clinical vignettes using structured templates
4. Outputs: data/processed/train.jsonl, data/processed/eval.jsonl
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from data.prompts.templates import (
    TRIAGE_VIGNETTE_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    EMERGENCY_TEMPLATE,
    format_sft_example,
)

# ===== Configuration =====
OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
REGULATIONS_DATASET = "Zolisa/cliniq-dataset"
EVAL_SPLIT_RATIO = 0.1
RANDOM_SEED = 42
MAX_SAMPLES_PER_SOURCE = 2000


# ===== Clinical Filter Keywords =====
CLINICAL_DOMAINS = {
    "tb": ["tuberculosis", "tb", "cough", "sputum", "night sweats", "weight loss"],
    "hiv": ["hiv", "aids", "antiretroviral", "art", "cd4", "viral load", "pmtct"],
    "malaria": ["malaria", "fever", "plasmodium", "antimalarial", "rdt"],
    "maternal": [
        "pregnant", "pregnancy", "antenatal", "postnatal", "delivery",
        "pre-eclampsia", "eclampsia", "bleeding", "labour",
    ],
    "paediatric": [
        "child", "infant", "baby", "newborn", "neonatal", "pediatric",
        "paediatric", "immunization", "growth", "diarrhea", "diarrhoea",
    ],
    "general": [
        "fever", "cough", "headache", "pain", "wound", "infection",
        "diabetes", "hypertension", "asthma", "pneumonia",
    ],
}

ALL_KEYWORDS = [kw for keywords in CLINICAL_DOMAINS.values() for kw in keywords]


def is_clinically_relevant(text: str) -> bool:
    """Check if a text contains primary-care-relevant clinical content."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in ALL_KEYWORDS)


# ===== Synthetic Vignette Generation =====

SAMPLE_VIGNETTES = [
    {
        "age": "34", "sex": "Female",
        "chief_complaint": "Persistent cough for 3 weeks with blood-tinged sputum",
        "duration": "3 weeks",
        "associated_symptoms": "Night sweats, weight loss of 5kg over past month, fatigue",
        "vital_signs": "Temp 37.8°C, HR 92, BP 110/70, RR 22, SpO2 95%",
        "history": "HIV-positive on ART for 2 years, household TB contact",
    },
    {
        "age": "28", "sex": "Female",
        "chief_complaint": "Severe headache and blurred vision at 36 weeks pregnant",
        "duration": "2 days",
        "associated_symptoms": "Swelling of hands and face, epigastric pain, reduced fetal movement",
        "vital_signs": "Temp 36.9°C, HR 98, BP 160/110, RR 18, SpO2 98%",
        "history": "First pregnancy, no prior hypertension, last antenatal visit 4 weeks ago",
    },
    {
        "age": "2", "sex": "Male",
        "chief_complaint": "High fever and refusing to eat for 2 days",
        "duration": "2 days",
        "associated_symptoms": "Watery diarrhea (6 episodes/day), sunken eyes, dry mouth, lethargy",
        "vital_signs": "Temp 39.2°C, HR 160, RR 40, capillary refill 3 seconds",
        "history": "Not vaccinated for rotavirus, no recent travel, breastfeeding stopped 1 month ago",
    },
    {
        "age": "45", "sex": "Male",
        "chief_complaint": "Recurring fever episodes every 48 hours",
        "duration": "1 week",
        "associated_symptoms": "Chills, rigors, profuse sweating, muscle aches, mild jaundice",
        "vital_signs": "Temp 40.1°C (during episode), HR 110, BP 100/60, RR 24",
        "history": "Lives in malaria-endemic area, no bed net, previous malaria episode 6 months ago",
    },
    {
        "age": "30", "sex": "Female",
        "chief_complaint": "Unintentional weight loss and persistent oral thrush",
        "duration": "3 months",
        "associated_symptoms": "Chronic diarrhea, recurrent skin infections, fatigue, lymphadenopathy",
        "vital_signs": "Temp 37.2°C, HR 88, BP 100/65, RR 16, SpO2 97%",
        "history": "Sexually active, multiple partners, never tested for HIV, no contraception use",
    },
    {
        "age": "8", "sex": "Female",
        "chief_complaint": "Fast breathing and chest indrawing",
        "duration": "3 days",
        "associated_symptoms": "Cough, fever, unable to drink fluids, nasal flaring",
        "vital_signs": "Temp 38.9°C, HR 140, RR 54, SpO2 89%",
        "history": "Incomplete vaccination schedule, recent measles in community, malnourished (WAZ -3)",
    },
    {
        "age": "55", "sex": "Male",
        "chief_complaint": "Persistent fatigue, excessive thirst and frequent urination",
        "duration": "2 months",
        "associated_symptoms": "Blurred vision, slow-healing wound on foot, tingling in feet",
        "vital_signs": "Temp 36.8°C, HR 82, BP 145/90, RR 16, BMI 32",
        "history": "Family history of diabetes, sedentary lifestyle, no previous screening",
    },
    {
        "age": "22", "sex": "Female",
        "chief_complaint": "Heavy vaginal bleeding 10 days after delivery",
        "duration": "6 hours",
        "associated_symptoms": "Dizziness, palpitations, soaking through pads every 30 minutes, mild fever",
        "vital_signs": "Temp 37.9°C, HR 120, BP 85/55, RR 26, pallor present",
        "history": "Home delivery by traditional birth attendant, prolonged labour, placenta appeared complete",
    },
]

# Corresponding clinical responses for the vignettes
SAMPLE_RESPONSES = [
    """**TRIAGE LEVEL: URGENT (Yellow)**
Justification: Productive cough >2 weeks with hemoptysis in an HIV-positive patient with TB contact requires urgent TB workup.

**KEY FINDINGS:**
- Cough >2 weeks with blood-tinged sputum (cardinal TB symptom)
- Constitutional symptoms: night sweats, significant weight loss (5kg/month)
- HIV-positive (increased TB risk despite ART)
- Known household TB contact exposure
- Low-grade fever, mildly tachypneic, borderline SpO2

**DIFFERENTIAL DIAGNOSES:**
1. Pulmonary Tuberculosis (most likely given symptom constellation + risk factors)
2. Bacterial pneumonia (community-acquired)
3. Pneumocystis pneumonia (PCP - if ART adherence is poor/low CD4)

**RECOMMENDED ACTIONS:**
- Collect 2 sputum samples (spot + early morning) for GeneXpert MTB/RIF
- Check most recent CD4 count and viral load
- Chest X-ray if available
- Assess ART adherence (check clinic records, pill count)
- Isolate patient (infection control measures)
- Do not delay: initiate workup TODAY

**REFERRAL DECISION:** Refer to TB focal point/doctor within 24 hours. If GeneXpert positive, initiate TB treatment urgently per national protocol. Contact tracing required for household members.

**SAFETY FLAGS:**
⚠️ Hemoptysis requires exclusion of severe/cavitary TB
⚠️ HIV-TB co-infection needs careful ART-TB drug interaction review
⚠️ Assess for TB drug resistance given prior contact history

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: EMERGENCY (Red)**
Justification: Severe pre-eclampsia at 36 weeks with dangerously elevated BP, neurological symptoms, and epigastric pain.

**KEY FINDINGS:**
- Severely elevated BP: 160/110 mmHg (threshold for severe pre-eclampsia)
- Neurological symptoms: severe headache + visual disturbance
- Epigastric pain (potential HELLP syndrome indicator)
- Generalized edema (face and hands)
- Reduced fetal movement (potential fetal compromise)
- Late third trimester (36 weeks - near term)

**DIFFERENTIAL DIAGNOSES:**
1. Severe pre-eclampsia (most likely - classic triad present)
2. HELLP syndrome (epigastric pain, needs urgent bloods)
3. Eclampsia (impending - headache and visual changes are warning signs)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: MgSO4 loading dose for seizure prophylaxis (4g IV over 20 min)
- IMMEDIATE: Antihypertensive (nifedipine 10mg PO or hydralazine 5mg IV)
- Continuous vital signs monitoring every 15 minutes
- Insert urinary catheter, monitor output (>30ml/hr target)
- Left lateral position
- Urgent blood: FBC, LFT, renal function, clotting
- Continuous fetal monitoring if available
- DO NOT DELAY TRANSFER

**REFERRAL DECISION:** EMERGENCY TRANSFER to nearest facility with C-section capability IMMEDIATELY. This patient needs delivery within hours. Call ahead to receiving facility.

**SAFETY FLAGS:**
🚨 LIFE-THREATENING: Imminent risk of eclamptic seizures
🚨 LIFE-THREATENING: Potential HELLP syndrome
🚨 Do NOT wait for blood results before initiating MgSO4 and transfer
🚨 Reduced fetal movement: fetal compromise likely

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: EMERGENCY (Red)**
Justification: 2-year-old with signs of severe dehydration (sunken eyes, dry mouth, lethargy) and high fever — IMCI danger signs present.

**KEY FINDINGS:**
- IMCI Danger Signs present: lethargy, unable to eat/drink
- Severe dehydration indicators: sunken eyes, dry mouth, prolonged capillary refill (3s)
- High-grade fever (39.2°C) with tachycardia (160) and tachypnea (40)
- 6 episodes watery diarrhea/day for 2 days
- Unvaccinated (rotavirus) — increases risk of severe gastroenteritis
- Recently stopped breastfeeding (loss of protective factors)

**DIFFERENTIAL DIAGNOSES:**
1. Acute gastroenteritis with severe dehydration (most likely - rotavirus probable)
2. Invasive bacterial enteritis (Shigella/Salmonella if bloody stool develops)
3. Sepsis secondary to gastroenteritis (tachycardia + lethargy concerning)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: Begin Plan C rehydration (IMCI protocol)
  - IV Ringer's Lactate/Normal Saline 30ml/kg over 1 hour, then 70ml/kg over 5 hours
  - If IV not possible: NG tube ORS 20ml/kg/hr
- Zinc supplementation: 20mg daily for 10-14 days
- Monitor urine output, skin turgor, and consciousness hourly
- Check blood glucose (hypoglycemia risk in malnourished child)
- Continue feeding once able to tolerate (small frequent amounts)
- Treat fever: Paracetamol 15mg/kg

**REFERRAL DECISION:** EMERGENCY TRANSFER if IV fluids cannot be given at this facility. Do not delay. Child needs supervised rehydration and monitoring.

**SAFETY FLAGS:**
🚨 IMCI DANGER SIGNS: Lethargy + inability to feed = SEVERE classification
🚨 Capillary refill >3s suggests circulatory compromise
🚨 Unvaccinated child at higher risk for complications
🚨 Monitor for convulsions (febrile seizure risk at this temperature)

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: URGENT (Yellow)**
Justification: Classic tertian malaria pattern with jaundice and hemodynamic instability suggesting moderate-severe malaria.

**KEY FINDINGS:**
- Classic 48-hour fever periodicity (P. vivax or P. falciparum pattern)
- High-grade fever (40.1°C) with rigors and profuse sweating
- Mild jaundice (hemolysis indicator)
- Tachycardia (110) and borderline hypotension (100/60)
- Endemic area, no bed net use, previous malaria episode
- 1-week duration increases risk of complications

**DIFFERENTIAL DIAGNOSES:**
1. Malaria — P. falciparum (most dangerous, must be excluded first)
2. Malaria — P. vivax (48-hour cycle fits, but jaundice less common)
3. Typhoid fever (endemic area, prolonged fever, but periodicity less typical)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: Malaria RDT (rapid diagnostic test) — if positive, treat immediately
- Blood smear (thick and thin film) for species identification and parasite density
- If RDT positive for P. falciparum: ACT (Artemether-Lumefantrine) per national protocol
- IV fluids for dehydration and hemodynamic support
- Monitor for severe malaria signs: altered consciousness, severe anemia, renal impairment
- Paracetamol for fever (avoid aspirin)
- Check hemoglobin — jaundice suggests hemolytic anemia

**REFERRAL DECISION:** Refer urgently if: parasite density >2%, hemoglobin <7g/dL, altered consciousness, persistent vomiting preventing oral medication, or no improvement within 24h of ACT.

**SAFETY FLAGS:**
⚠️ Jaundice + hemodynamic instability = potential severe malaria
⚠️ Previous malaria does NOT confer full immunity
⚠️ 1-week delay increases risk of cerebral malaria and organ damage
⚠️ Counsel on bed net use and malaria prophylaxis for household

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: URGENT (Yellow)**
Justification: Clinical presentation highly suggestive of undiagnosed HIV/AIDS with WHO Stage 3-4 indicators. Requires urgent testing and staging.

**KEY FINDINGS:**
- Chronic weight loss (3 months unintentional)
- Persistent oral thrush (candidiasis — WHO Clinical Stage 3)
- Chronic diarrhea >1 month (WHO Stage 3)
- Recurrent skin infections
- Generalized lymphadenopathy
- High-risk behavior: multiple sexual partners, no condom use
- Never previously tested

**DIFFERENTIAL DIAGNOSES:**
1. HIV/AIDS — advanced disease (Stage 3-4 based on clinical features)
2. Chronic immunosuppression — other cause (less likely given risk profile)
3. Lymphoma (consider if HIV test negative — weight loss, lymphadenopathy)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: HIV rapid test (with pre-test counseling)
- If positive: confirmatory test per national algorithm
- Baseline bloods: CD4 count, viral load, FBC, renal and liver function
- Screen for TB (sputum GeneXpert — oral thrush + weight loss = high TB risk)
- Screen for cryptococcal antigen if CD4 <200
- Initiate ART as soon as possible (same-day initiation per current WHO guidelines)
- Nutritional assessment and support
- STI screening and treatment
- Partner notification and testing counseling

**REFERRAL DECISION:** Referral to ART initiation site if not available at this facility. Urgent, not emergency — initiate within 7 days. Same-day if CD4 <200.

**SAFETY FLAGS:**
⚠️ Oral thrush + weight loss + diarrhea = presumptive advanced HIV — do NOT wait for CD4 before referring
⚠️ High TB co-infection risk — concurrent screening mandatory
⚠️ Cryptococcal meningitis risk if severely immunosuppressed
⚠️ Provide post-test counseling regardless of result

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: EMERGENCY (Red)**
Justification: 8-year-old with IMCI danger signs: unable to drink, severe tachypnea (RR 54), chest indrawing, and hypoxia (SpO2 89%). Severe pneumonia classification.

**KEY FINDINGS:**
- IMCI Danger Signs: unable to drink, chest indrawing
- Severe respiratory distress: RR 54 (>40 for age = fast breathing), nasal flaring
- Hypoxia: SpO2 89% (critical — needs oxygen)
- Fever with cough for 3 days (lower respiratory tract infection)
- Severe malnutrition (WAZ -3) — extreme risk factor for mortality
- Incomplete vaccination and recent community measles exposure
- Post-measles pneumonia carries very high mortality

**DIFFERENTIAL DIAGNOSES:**
1. Severe pneumonia — bacterial (most likely, IMCI classification guides treatment)
2. Measles pneumonia (community outbreak exposure, incomplete vaccination)
3. Pulmonary TB (malnourished child, but acute presentation favors pneumonia)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: Oxygen therapy if available (target SpO2 >92%)
- IMMEDIATE: First dose antibiotics — IV Ampicillin + Gentamicin (IMCI severe pneumonia)
- If IV not possible: IM Ceftriaxone 50mg/kg as single dose before transfer
- Position: prop upright, keep warm (hypothermia risk in malnourished child)
- Check blood glucose urgently (10% dextrose if <3 mmol/L)
- Nasogastric feeds if cannot swallow (F-75 therapeutic milk for SAM)
- DO NOT DELAY TRANSFER

**REFERRAL DECISION:** EMERGENCY TRANSFER IMMEDIATELY to hospital with oxygen, IV antibiotics, and inpatient capacity. This child is at extreme mortality risk.

**SAFETY FLAGS:**
🚨 SpO2 89% — life-threatening hypoxia, OXYGEN IS PRIORITY ONE
🚨 Severe malnutrition (WAZ -3) — dramatically increases pneumonia mortality
🚨 Incomplete vaccination + measles exposure — empiric vitamin A (200,000 IU)
🚨 High risk of hypoglycemia, hypothermia, and septic shock

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: URGENT (Yellow)**
Justification: Classic presentation of undiagnosed Type 2 Diabetes Mellitus with early complications (neuropathy, poor wound healing). Requires same-day diagnosis and management initiation.

**KEY FINDINGS:**
- Classic diabetes triad: polyuria, polydipsia, fatigue
- Duration 2 months (suggests chronic uncontrolled hyperglycemia)
- Early complications already present: peripheral neuropathy (tingling), retinopathy (blurred vision), poor wound healing
- Major risk factors: obesity (BMI 32), family history, sedentary lifestyle
- Hypertension (145/90) — metabolic syndrome likely
- Non-healing foot wound (diabetic foot risk — urgent attention)

**DIFFERENTIAL DIAGNOSES:**
1. Type 2 Diabetes Mellitus (most likely — classic presentation + risk factors)
2. Type 1 Diabetes / LADA (less likely at 55, but check if very lean/ketotic)
3. Diabetes secondary to other endocrine disease (Cushing's — less likely)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: Fasting blood glucose or random blood glucose + HbA1c
- Urine dipstick: glucose, ketones, protein (check for ketosis and nephropathy)
- Examine foot wound: clean, dress, assess for infection (antibiotics if cellulitis)
- Check creatinine and urine protein (diabetic nephropathy screening)
- Fundoscopy referral (blurred vision — diabetic retinopathy screening)
- Initiate: Metformin 500mg BD (if eGFR >30, no contraindications)
- Antihypertensive: lifestyle + ACE inhibitor (renoprotective in diabetes)
- Lifestyle counseling: diet, exercise, weight management
- Diabetic foot care education

**REFERRAL DECISION:** Refer for ophthalmology review (blurred vision) within 2 weeks. If foot wound shows spreading infection, refer within 24 hours.

**SAFETY FLAGS:**
⚠️ Non-healing foot wound = high amputation risk if neglected
⚠️ Blurred vision may indicate diabetic retinopathy — sight-threatening
⚠️ Check for ketones — if present, consider urgent referral (DKA risk)
⚠️ Dual pathology: diabetes + hypertension = accelerated cardiovascular risk

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",

    """**TRIAGE LEVEL: EMERGENCY (Red)**
Justification: Postpartum hemorrhage (PPH) with hemodynamic instability — life-threatening obstetric emergency requiring immediate intervention.

**KEY FINDINGS:**
- Active heavy vaginal bleeding 10 days postpartum (secondary PPH)
- Hemodynamic compromise: tachycardia (120), hypotension (85/55), pallor
- Soaking pads every 30 minutes = estimated blood loss >500ml and ongoing
- Low-grade fever (37.9°C) suggests possible endometritis/retained products
- Risk factors: prolonged labour, home delivery without skilled birth attendant
- Placenta "appeared complete" but retained fragments cannot be excluded

**DIFFERENTIAL DIAGNOSES:**
1. Secondary PPH — retained placental fragments (most likely given timing and history)
2. Postpartum endometritis with hemorrhage (fever supports infection)
3. Uterine subinvolution (atonic uterus)

**RECOMMENDED ACTIONS:**
- IMMEDIATE: IV access x2 large bore cannulae, IV crystalloid bolus 1-2L
- IMMEDIATE: Bimanual uterine compression if uterus is atonic/boggy
- Oxytocin 10 IU IM + 20 IU in 1L saline IV infusion
- Misoprostol 800mcg sublingual if oxytocin unavailable
- Cross-match blood / arrange transfusion
- IV antibiotics: Ampicillin + Gentamicin + Metronidazole (cover endometritis)
- Keep patient warm, elevate legs (shock management)
- Foley catheter to monitor urine output (>30ml/hr target)
- DO NOT ATTEMPT MANUAL EXPLORATION — requires surgical facility

**REFERRAL DECISION:** EMERGENCY TRANSFER IMMEDIATELY to facility with blood transfusion, ultrasound, and surgical/evacuation capability. Call ahead. Patient may need examination under anesthesia + evacuation of retained products.

**SAFETY FLAGS:**
🚨 LIFE-THREATENING: Active hemorrhage with shock (tachycardia + hypotension)
🚨 Estimated ongoing blood loss >500ml — will deteriorate rapidly
🚨 Do NOT wait for ultrasound confirmation — treat as retained products until proven otherwise
🚨 Fever = likely concurrent infection — dual life-threat (sepsis + hemorrhage)

*Disclaimer: This assessment is for decision-support only. Clinical confirmation by a qualified healthcare professional is required.*""",
]


def generate_synthetic_vignettes() -> list[dict]:
    """
    Generate training examples from structured clinical vignette templates.
    
    Returns:
        list[dict]: Training examples with 'messages' field.
    """
    examples = []

    for vignette, response in zip(SAMPLE_VIGNETTES, SAMPLE_RESPONSES):
        query = TRIAGE_VIGNETTE_TEMPLATE.format(**vignette)
        messages = format_sft_example(query, response)
        examples.append({"messages": messages})

    return examples


def load_and_filter_medical_qa() -> list[dict]:
    """
    Load MedQA-style datasets from HuggingFace and filter for clinical relevance.
    
    Falls back to synthetic data if datasets are unavailable.
    
    Returns:
        list[dict]: Filtered training examples.
    """
    examples = []

    # Try loading HealthCareMagic subset
    try:
        print("📥 Loading medical QA dataset...")
        dataset = load_dataset(
            "lavita/ChatDoctor-HealthCareMagic-100k",
            split="train",
            trust_remote_code=True,
        )

        count = 0
        for item in dataset:
            if count >= MAX_SAMPLES_PER_SOURCE:
                break

            instruction = item.get("instruction", item.get("input", ""))
            output = item.get("output", item.get("response", ""))

            if instruction and output and is_clinically_relevant(instruction):
                messages = format_sft_example(instruction, output)
                examples.append({"messages": messages})
                count += 1

        print(f"✅ Loaded {count} examples from HealthCareMagic")

    except Exception as e:
        print(f"⚠️ Could not load HealthCareMagic dataset: {e}")
        print("   Falling back to synthetic data only.")

    return examples


def load_regulatory_data() -> list[dict]:
    """
    Load regulatory/compliance data from HuggingFace dataset.
    
    Returns:
        list[dict]: Training examples from regulations dataset.
    """
    examples = []
    try:
        print(f"   Loading from {REGULATIONS_DATASET}...")
        ds = load_dataset(REGULATIONS_DATASET, split="train")
        for item in ds:
            # Format as Q&A for training
            query = f"Explain the key requirements of {item['source'].upper()} for healthcare data handling."
            response = item["content"]
            messages = format_sft_example(query, response)
            examples.append({"messages": messages})
        print(f"   Loaded {len(examples)} regulatory examples")
    except Exception as e:
        print(f"   ⚠️ Could not load regulations: {e}")
    return examples


def prepare_dataset():
    """
    Main data preparation pipeline.
    
    Combines synthetic vignettes, filtered open datasets,
    and regulatory data, then shuffles and splits into train/eval.
    """
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ClinIQ Data Preparation Pipeline")
    print("=" * 60)

    # 1. Generate synthetic clinical vignettes
    print("\n📝 Generating synthetic clinical vignettes...")
    synthetic = generate_synthetic_vignettes()
    print(f"   Generated {len(synthetic)} synthetic vignettes")

    # 2. Load and filter open medical QA datasets
    print("\n📥 Loading open medical QA datasets...")
    medical_qa = load_and_filter_medical_qa()
    print(f"   Loaded {len(medical_qa)} filtered medical QA examples")

    # 3. Load regulatory/compliance data
    print("\n📜 Loading regulatory datasets...")
    regulations = load_regulatory_data()
    print(f"   Loaded {len(regulations)} regulatory examples")

    # 4. Combine and shuffle
    all_examples = synthetic + medical_qa + regulations
    random.shuffle(all_examples)
    print(f"\n📊 Total examples: {len(all_examples)}")

    # 4. Split into train/eval
    split_idx = max(1, int(len(all_examples) * (1 - EVAL_SPLIT_RATIO)))
    train_data = all_examples[:split_idx]
    eval_data = all_examples[split_idx:]

    # 5. Save as JSONL
    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path = OUTPUT_DIR / "eval.jsonl"

    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(train_data)} training examples to {train_path}")
    print(f"✅ Saved {len(eval_data)} evaluation examples to {eval_path}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()
