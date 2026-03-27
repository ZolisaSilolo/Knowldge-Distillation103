"""
regulations.py — Regulatory and compliance data for clinical training.
"""

REGULATORY_SOURCES = {
    "popia": {"name": "POPIA", "url": "https://popia.co.za/"},
    "hipaa": {"name": "HIPAA", "url": "https://www.hhs.gov/hipaa/index.html"},
    "fhir_r4": {"name": "FHIR R4", "url": "https://hl7.org/FHIR/R4/"},
    "sa_national_health_act": {"name": "National Health Act 2003", "url": "https://www.gov.za/about-SA/Health"},
    "sa_stg_eml": {"name": "STGs & EML", "url": "https://knowledgehub.health.gov.za/content/standard-treatment-guidelines-and-essential-medicines-list"},
}

COMPLIANCE_REQUIREMENTS = {
    "popia": [
        "Health data is special personal information requiring explicit consent",
        "Data minimization: collect only necessary for triage",
        "Secure storage and transmission of patient data",
    ],
    "hipaa": [
        "PHI protected with administrative, physical, technical safeguards",
        "Business Associate Agreements required for third-party handling",
        "Patient consent required for PHI disclosure",
    ],
    "fhir_r4": [
        "Use Patient, Encounter, Observation, Condition resources",
        "RESTful API standards for interoperability",
    ],
    "sa_national_health_act": [
        "Public and private health systems operate in parallel",
        "National Department of Health sets framework",
    ],
}
