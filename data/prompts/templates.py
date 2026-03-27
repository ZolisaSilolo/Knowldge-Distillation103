"""
templates.py — Chat template formatting for Qwen2.5 models.

Converts clinical vignettes into properly formatted chat-template messages
for both SFT (Stages C/B) and distillation (Stage A) training.
"""

from pathlib import Path

# Load system prompt at import time
_PROMPTS_DIR = Path(__file__).resolve().parent
_SYSTEM_PROMPT = (_PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()


def get_system_prompt() -> str:
    """Return the ClinIQ clinical system prompt."""
    return _SYSTEM_PROMPT


def format_sft_example(patient_query: str, clinical_response: str) -> list[dict]:
    """
    Format a single training example as a Qwen2.5 chat message list.
    
    Used in Stages C and B for supervised fine-tuning.
    
    Args:
        patient_query: The patient's symptom description / health worker's question.
        clinical_response: The ideal clinical triage response.
        
    Returns:
        list[dict]: Chat messages in OpenAI-compatible format.
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": patient_query},
        {"role": "assistant", "content": clinical_response},
    ]


def format_inference_prompt(patient_query: str) -> list[dict]:
    """
    Format a patient query for inference (no assistant response).
    
    Used during evaluation and deployment.
    
    Args:
        patient_query: The symptom description or clinical question.
        
    Returns:
        list[dict]: Chat messages ready for model.generate().
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": patient_query},
    ]


def format_distillation_example(
    patient_query: str,
    teacher_response: str,
    teacher_logits_path: str = None,
) -> dict:
    """
    Format a training example for Stage A (KL distillation).
    
    Includes the teacher's response and optional path to pre-computed logits.
    
    Args:
        patient_query: The patient's symptom description.
        teacher_response: The teacher model's generated response.
        teacher_logits_path: Optional path to saved teacher logit tensors.
        
    Returns:
        dict: Training example with messages and metadata.
    """
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": patient_query},
            {"role": "assistant", "content": teacher_response},
        ],
        "teacher_logits_path": teacher_logits_path,
    }


# ===== Clinical Vignette Templates =====
# Used by data/prepare.py to generate synthetic training examples

TRIAGE_VIGNETTE_TEMPLATE = """Patient presents at the clinic with the following:
- Age: {age}, Sex: {sex}
- Chief Complaint: {chief_complaint}
- Duration: {duration}
- Associated Symptoms: {associated_symptoms}
- Vital Signs: {vital_signs}
- Relevant History: {history}

Please assess this patient and provide your triage recommendation."""

FOLLOW_UP_TEMPLATE = """Patient returning for follow-up:
- Age: {age}, Sex: {sex}
- Original Diagnosis: {original_diagnosis}
- Current Treatment: {current_treatment}
- Current Status: {current_status}
- Concerns: {concerns}

Please review and advise on next steps."""

EMERGENCY_TEMPLATE = """URGENT: Community health worker reports:
- Patient: {age} year old {sex}
- Situation: {situation}
- Current State: {current_state}
- Available Resources: {resources}
- Distance to Nearest Hospital: {distance}

Provide immediate triage and action steps."""
