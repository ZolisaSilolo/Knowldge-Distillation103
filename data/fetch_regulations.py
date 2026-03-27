"""
fetch_regulations.py — Fetch regulatory content and upload to HuggingFace.
"""

import json
import os
from pathlib import Path

# Regulatory content (manually curated for training)
REGULATORY_CONTENT = {
    "popia": """
POPIA - Protection of Personal Information Act (South Africa)
Health data is considered 'special personal information' requiring explicit consent.
Requirements:
- Lawful processing: must have a lawful purpose
- Purpose specification: clearly state why data is collected
- Minimality: only collect what is necessary
- Storage limitation: don't keep longer than needed
- Integrity: keep data accurate and secure
- Confidentiality: protect against unauthorized access
Patient rights: access, correction, deletion, objection.
""",
    "hipaa": """
HIPAA - Health Insurance Portability and Accountability Act (US)
Protected Health Information (PHI) includes any individually identifiable health information.
Privacy Rule: protects all individually identifiable health information held or transmitted.
Security Rule: requires administrative, physical, and technical safeguards.
Breach Notification Rule: notify affected individuals within 60 days.
Requirements for healthcare AI:
- Business Associate Agreements (BAA) required
- Minimum necessary standard
- Patient authorization for disclosure
- Audit trails and access logs
""",
    "fhir_r4": """
FHIR R4 - Fast Healthcare Interoperability Resources v4.0.1
Standard for exchanging healthcare information electronically.
Core resources: Patient, Encounter, Observation, Condition, MedicationRequest, DiagnosticReport.
RESTful API: GET/POST/PUT/DELETE operations on resources.
Data formats: JSON (preferred), XML.
Use for: EHR interoperability, patient data exchange, clinical research.
""",
    "sa_national_health_act": """
National Health Act 2003 (South Africa)
Provides framework for structured and uniform health system.
Key provisions:
- National, provincial, municipal health departments
- Public health services must provide access
- Private sector regulation
- Health research requirements
- Patient rights in public facilities
""",
    "sa_stg_eml": """
South Africa Standard Treatment Guidelines (STG) and Essential Medicines List (EML)
Primary healthcare standard treatment guidelines.
Key sections:
- Adult STGs
- Paediatric STGs  
- Essential Medicines List
- TB treatment protocols
- HIV/ART guidelines
- Maternal health protocols
""",
}


def main():
    # Save locally first
    output_dir = Path(__file__).parent / "processed"
    output_dir.mkdir(exist_ok=True)
    
    for name, content in REGULATORY_CONTENT.items():
        path = output_dir / f"regulations_{name}.txt"
        path.write_text(content.strip())
        print(f"✅ Saved {path}")
    
    # Create combined dataset
    combined = [{"source": k, "content": v.strip()} for k, v in REGULATORY_CONTENT.items()]
    
    with open(output_dir / "regulations.jsonl", "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Saved regulations.jsonl to {output_dir}")
    print("\nTo upload to HuggingFace:")
    print("  huggingface-cli upload <repo-id> data/processed/regulations.jsonl")


if __name__ == "__main__":
    main()
