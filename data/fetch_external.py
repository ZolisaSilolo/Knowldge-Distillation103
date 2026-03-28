"""
fetch_external.py — Fetch real clinical datasets from HuggingFace Hub.

Sources:
  1. intronhealth/afrimedqa_v2       — Africa-specific medical QA (CC BY-NC-SA 4.0)
  2. GBaker/MedQA-USMLE-4-options   — USMLE clinical vignettes (MIT)
  3. openlifescienceai/medmcqa       — Indian medical entrance QA (MIT)
  4. bigbio/meddialog                — Real doctor-patient dialogues (Apache 2.0)
  5. lavita/ChatDoctor-HealthCareMagic-100k — Patient-doctor chat (research use)
"""

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
MAX_PER_SOURCE = 2000

CLINICAL_KEYWORDS = [
    "fever", "cough", "pain", "infection", "pregnant", "child", "infant",
    "hiv", "tb", "tuberculosis", "malaria", "diarrhea", "diarrhoea",
    "bleeding", "breathing", "vomit", "rash", "wound", "diabetes",
    "hypertension", "anemia", "anaemia", "referral", "emergency", "urgent",
]


def is_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CLINICAL_KEYWORDS)


def to_messages(query: str, response: str) -> dict:
    return {"messages": [
        {"role": "user", "content": query.strip()},
        {"role": "assistant", "content": response.strip()},
    ]}


def fetch_afrimedqa() -> list[dict]:
    """Africa-specific medical QA — most relevant to ClinIQ's target context."""
    print("  📥 AfriMedQA v2 (intronhealth/afrimedqa_v2)...")
    try:
        ds = load_dataset("intronhealth/afrimedqa_v2", split="train", trust_remote_code=True)
        examples = []
        for item in ds:
            q = item.get("question", "")
            a = item.get("answer", item.get("explanation", ""))
            if q and a:
                examples.append(to_messages(q, a))
            if len(examples) >= MAX_PER_SOURCE:
                break
        print(f"     ✅ {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        return []


def fetch_medqa_usmle() -> list[dict]:
    """USMLE clinical vignettes — high-quality structured case presentations."""
    print("  📥 MedQA-USMLE (GBaker/MedQA-USMLE-4-options)...")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", trust_remote_code=True)
        examples = []
        for item in ds:
            q = item.get("question", "")
            # Build answer from options + correct answer
            options = item.get("options", {})
            answer_key = item.get("answer_idx", item.get("answer", ""))
            answer_text = options.get(answer_key, answer_key) if isinstance(options, dict) else str(answer_key)
            if q and answer_text and is_relevant(q):
                full_answer = f"Answer: {answer_text}"
                if isinstance(options, dict):
                    opts = "\n".join(f"  {k}: {v}" for k, v in options.items())
                    full_answer = f"Options:\n{opts}\n\nCorrect Answer: {answer_text}"
                examples.append(to_messages(q, full_answer))
            if len(examples) >= MAX_PER_SOURCE:
                break
        print(f"     ✅ {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        return []


def fetch_medmcqa() -> list[dict]:
    """Indian medical entrance QA — broad primary care coverage."""
    print("  📥 MedMCQA (openlifescienceai/medmcqa)...")
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split="train", trust_remote_code=True)
        option_map = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
        examples = []
        for item in ds:
            q = item.get("question", "")
            correct_idx = item.get("cop", 0)
            correct_key = option_map.get(correct_idx, "opa")
            answer = item.get(correct_key, "")
            explanation = item.get("exp", "")
            if q and answer and is_relevant(q):
                response = answer
                if explanation:
                    response = f"{answer}\n\nExplanation: {explanation}"
                examples.append(to_messages(q, response))
            if len(examples) >= MAX_PER_SOURCE:
                break
        print(f"     ✅ {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        return []


def fetch_meddialog() -> list[dict]:
    """Real doctor-patient dialogues from HealthCareMagic and iCliniq."""
    print("  📥 MedDialog (bigbio/meddialog)...")
    try:
        ds = load_dataset("bigbio/meddialog", "meddialog_en_bigbio_qa", split="train", trust_remote_code=True)
        examples = []
        for item in ds:
            q = item.get("question", "")
            a = " ".join(item.get("answer", [])) if isinstance(item.get("answer"), list) else item.get("answer", "")
            if q and a and is_relevant(q):
                examples.append(to_messages(q, a))
            if len(examples) >= MAX_PER_SOURCE:
                break
        print(f"     ✅ {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        return []


def fetch_healthcaremagic() -> list[dict]:
    """Patient-doctor chat from HealthCareMagic — broad primary care."""
    print("  📥 ChatDoctor-HealthCareMagic (lavita/ChatDoctor-HealthCareMagic-100k)...")
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", trust_remote_code=True)
        examples = []
        for item in ds:
            q = item.get("instruction", item.get("input", ""))
            a = item.get("output", item.get("response", ""))
            if q and a and is_relevant(q):
                examples.append(to_messages(q, a))
            if len(examples) >= MAX_PER_SOURCE:
                break
        print(f"     ✅ {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        return []


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("🌐 Fetching real clinical datasets...\n")

    all_examples = []
    all_examples += fetch_afrimedqa()
    all_examples += fetch_medqa_usmle()
    all_examples += fetch_medmcqa()
    all_examples += fetch_meddialog()
    all_examples += fetch_healthcaremagic()

    output_path = OUTPUT_DIR / "external_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Total: {len(all_examples)} examples → {output_path}")
    return all_examples


if __name__ == "__main__":
    main()
