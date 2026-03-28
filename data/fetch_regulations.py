"""
fetch_regulations.py — Fetch real regulatory/guideline content from official sources
and upload to HuggingFace as a dataset.

Sources:
  - WHO IMAI/IMCI guidelines (PDF text via WHO website)
  - SA STG/EML (knowledgehub.health.gov.za)
  - HIPAA summary (hhs.gov)
  - POPIA full text (gov.za)
  - FHIR R4 resource definitions (hl7.org)
"""

import json
import os
import re
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"

# Official sources — plain text or HTML that can be scraped without auth
SOURCES = [
    {
        "key": "who_imai",
        "name": "WHO IMAI District Clinician Manual",
        "url": "https://www.who.int/publications/i/item/9789241548038",
        "fallback_text": """WHO Integrated Management of Adult Illness (IMAI) District Clinician Manual.
Key triage principles:
- Assess all patients for emergency signs first (airway, breathing, circulation, consciousness)
- Emergency signs require immediate treatment before history-taking
- Priority signs: very fast/slow breathing, severe pallor, restlessness, high fever, severe dehydration
- Triage levels: Emergency (immediate) → Priority (seen next) → Non-urgent (queue)
- HIV/TB co-infection: screen all TB patients for HIV; screen all HIV patients for TB
- ART: initiate regardless of CD4 count; do not stop during acute illness
- Malaria: RDT first; treat confirmed cases with ACT per national protocol
- Sepsis: IV antibiotics within 1 hour of recognition; fluid resuscitation
- Referral: document reason, stabilise before transfer, communicate with receiving facility""",
    },
    {
        "key": "who_imci",
        "name": "WHO IMCI Integrated Management of Childhood Illness",
        "url": "https://www.who.int/publications/i/item/9789240018372",
        "fallback_text": """WHO Integrated Management of Childhood Illness (IMCI) Guidelines.
Danger signs in children (any = emergency referral):
- Unable to drink or breastfeed
- Vomits everything
- Convulsions (current or history)
- Lethargic or unconscious
- Stridor in calm child
- Severe malnutrition (visible wasting, oedema of both feet)
Classification system:
- Cough/breathing: pneumonia (fast breathing) → severe pneumonia (chest indrawing, SpO2<90%)
- Diarrhoea: some dehydration (Plan B ORS) → severe dehydration (Plan C IV fluids)
- Fever: malaria RDT → treat if positive; meningitis signs → emergency referral
- Malnutrition: MUAC <115mm = SAM → therapeutic feeding + antibiotics
- HIV: test at 6 weeks; start ART if positive regardless of CD4
Treatments:
- Pneumonia: Amoxicillin 40mg/kg/day x5 days (mild); IV Ampicillin+Gentamicin (severe)
- Dehydration: ORS 75ml/kg over 4h (Plan B); Ringer's Lactate 100ml/kg (Plan C)
- Malaria: ACT per national protocol; severe malaria IV Artesunate
- SAM: F-75 stabilisation → F-100 rehabilitation → RUTF""",
    },
    {
        "key": "sa_stg_primary",
        "name": "South Africa Primary Healthcare Standard Treatment Guidelines",
        "url": "https://knowledgehub.health.gov.za/elibrary/primary-healthcare-standard-treatment-guidelines-and-essential-medicines-list-2020",
        "fallback_text": """South Africa Primary Healthcare Standard Treatment Guidelines (2020 Edition).
HIV/ART:
- First line: TDF 300mg + 3TC 300mg + DTG 50mg (once daily)
- Pregnant women: same first-line regimen; start immediately
- Children <25kg: ABC + 3TC + LPV/r
- Viral load monitoring: 6 months after initiation, then annually
- Adherence counselling at every visit
Tuberculosis:
- Drug-sensitive TB: 2RHZE/4RH (6 months total)
- TB preventive therapy (TPT): Isoniazid 300mg daily x6 months for HIV-positive contacts
- GeneXpert MTB/RIF: first-line diagnostic for all presumptive TB
- Drug-resistant TB: refer to DR-TB specialist
Malaria (endemic areas):
- Uncomplicated: Artemether-Lumefantrine (AL) x3 days
- Severe: IV Artesunate; refer to hospital
Maternal Health:
- Antenatal: iron/folate, TT vaccine, HIV test, syphilis screen, BP monitoring
- Pre-eclampsia: MgSO4 + antihypertensive + urgent referral
- PPH: oxytocin 10IU IM immediately after delivery; misoprostol if unavailable
Paediatric:
- Diarrhoea: ORS + zinc 20mg x10-14 days; IV fluids if severe
- Pneumonia: Amoxicillin 40mg/kg/day; refer if severe
- Malnutrition: MUAC screening; refer SAM (<115mm or oedema)""",
    },
    {
        "key": "hipaa_summary",
        "name": "HIPAA Privacy and Security Rules Summary",
        "url": "https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html",
        "fallback_text": """HIPAA Privacy Rule and Security Rule — Key Requirements for Healthcare AI.
Protected Health Information (PHI):
- Any individually identifiable health information in any form (electronic, paper, oral)
- Includes: name, address, dates, phone, SSN, medical record numbers, diagnoses, treatments
Privacy Rule requirements:
- Minimum necessary standard: disclose only what is needed for the purpose
- Patient rights: access, amendment, accounting of disclosures, restriction requests
- Permitted uses: treatment, payment, healthcare operations (no authorisation needed)
- Required authorisation: marketing, sale of PHI, most research uses
Security Rule (ePHI):
- Administrative safeguards: risk analysis, workforce training, access management
- Physical safeguards: facility access controls, workstation security, device controls
- Technical safeguards: access controls, audit controls, integrity controls, transmission security
- Encryption: addressable (strongly recommended) for data at rest and in transit
AI/ML specific:
- Business Associate Agreement (BAA) required for any vendor processing PHI
- De-identification: Safe Harbor (18 identifiers removed) or Expert Determination method
- Model training on PHI requires patient authorisation or IRB waiver
Breach notification:
- Notify affected individuals within 60 days of discovery
- Notify HHS; notify media if >500 individuals in a state""",
    },
    {
        "key": "popia",
        "name": "POPIA Protection of Personal Information Act South Africa",
        "url": "https://www.justice.gov.za/inforeg/docs/InfoRegSA-POPIA-act4of2013.pdf",
        "fallback_text": """POPIA — Protection of Personal Information Act 4 of 2013 (South Africa).
Health information is 'special personal information' — highest protection category.
Eight conditions for lawful processing:
1. Accountability: responsible party must ensure compliance
2. Processing limitation: lawful, with consent or legitimate purpose
3. Purpose specification: collected for specific, explicitly defined purpose
4. Further processing limitation: compatible with original purpose
5. Information quality: accurate, complete, not misleading
6. Openness: notify data subject of collection
7. Security safeguards: technical and organisational measures
8. Data subject participation: right to access, correction, deletion
Health data specific rules:
- Explicit consent required for processing health information
- Processing permitted without consent: medical treatment, insurance, research (with conditions)
- Cross-border transfer: only to countries with adequate protection or with consent
- Data subject rights: access within 30 days, correction, objection, deletion
- Breach notification: notify Information Regulator and data subjects without delay
Penalties:
- Administrative fines up to R10 million
- Criminal penalties: imprisonment up to 10 years for serious offences
Relevance to ClinIQ:
- Patient symptom data = personal information (possibly special)
- Offline processing (no transmission) reduces risk surface
- No patient identifiers should be stored in model inputs/outputs""",
    },
    {
        "key": "fhir_r4",
        "name": "HL7 FHIR R4 Clinical Resources Reference",
        "url": "https://hl7.org/fhir/R4/resourcelist.html",
        "fallback_text": """HL7 FHIR R4 — Fast Healthcare Interoperability Resources v4.0.1.
Core clinical resources:
- Patient: demographics, identifiers, contact information
- Encounter: clinical visit, triage classification, disposition
- Observation: vital signs (BP, HR, RR, Temp, SpO2, weight, height)
- Condition: diagnosis, problem list, clinical findings
- MedicationRequest: prescriptions, dosing, route, frequency
- DiagnosticReport: lab results, imaging reports, GeneXpert results
- Procedure: clinical interventions performed
- AllergyIntolerance: drug and food allergies
- Immunization: vaccination records
Triage-relevant resources:
- Encounter.priority: emergency | urgent | routine (maps to ClinIQ triage levels)
- Encounter.reasonCode: chief complaint coding (SNOMED CT)
- Observation.code: LOINC codes for vital signs
  - 8310-5: Body temperature
  - 8867-4: Heart rate
  - 9279-1: Respiratory rate
  - 55284-4: Blood pressure
  - 59408-5: SpO2
RESTful API patterns:
- GET /Patient/{id} — retrieve patient record
- POST /Encounter — create new encounter
- PUT /Observation — update vital signs
- GET /Condition?patient={id} — list patient conditions
Data formats: JSON (preferred for mobile/offline), XML
Offline use: FHIR resources can be stored locally as JSON files without server""",
    },
]


def fetch_url_text(url: str, timeout: int = 10) -> str | None:
    """Attempt to fetch plain text from a URL."""
    try:
        req = Request(url, headers={"User-Agent": "ClinIQ-DataPipeline/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:8000] if len(text) > 8000 else text
    except (URLError, HTTPError):
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []

    print("📜 Fetching regulatory/guideline content...\n")

    for source in SOURCES:
        print(f"  {source['name']}...")
        content = fetch_url_text(source["url"])

        if content and len(content) > 500:
            print(f"     ✅ Fetched {len(content)} chars from {source['url']}")
        else:
            print(f"     ⚠️  URL not directly scrapable — using curated fallback text")
            content = source["fallback_text"].strip()

        # Save individual file
        txt_path = OUTPUT_DIR / f"regulations_{source['key']}.txt"
        txt_path.write_text(content, encoding="utf-8")

        combined.append({"source": source["key"], "name": source["name"], "content": content})
        time.sleep(0.5)  # polite crawl delay

    # Save combined JSONL
    jsonl_path = OUTPUT_DIR / "regulations.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(combined)} regulatory documents → {jsonl_path}")

    # Upload to HuggingFace if token available
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REGULATIONS_REPO", "Zolisa/cliniq-dataset")

    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(jsonl_path),
                path_in_repo="regulations.jsonl",
                repo_id=hf_repo,
                repo_type="dataset",
                token=hf_token,
            )
            print(f"✅ Uploaded to HuggingFace: {hf_repo}")
        except Exception as e:
            print(f"⚠️  HuggingFace upload failed: {e}")
    else:
        print("⚠️  HF_TOKEN not set — skipping upload")
        print(f"   Run: huggingface-cli upload {hf_repo} data/processed/regulations.jsonl --repo-type dataset")

    return combined


if __name__ == "__main__":
    main()
