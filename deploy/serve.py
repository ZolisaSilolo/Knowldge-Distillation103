"""
serve.py — FastAPI local inference server wrapping Ollama.

Provides a structured REST API for the ClinIQ distilled model,
running entirely offline via Ollama on commodity hardware.

Usage:
    python deploy/serve.py
    # API available at http://localhost:8000
    # Docs at http://localhost:8000/docs
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ===== Configuration =====
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("CLINIQ_MODEL", "cliniq")
SERVE_HOST = os.environ.get("SERVE_HOST", "0.0.0.0")
SERVE_PORT = int(os.environ.get("SERVE_PORT", "8000"))


# ===== API Models =====

class TriageRequest(BaseModel):
    """Patient symptom input for triage assessment."""
    patient_description: str = Field(
        ...,
        description="Patient symptoms, vital signs, and relevant history.",
        min_length=10,
        examples=[
            "34-year-old female with persistent cough for 3 weeks, "
            "blood-tinged sputum, night sweats, weight loss. "
            "HIV-positive on ART. Household TB contact."
        ],
    )
    urgency_context: Optional[str] = Field(
        None,
        description="Additional context: 'community', 'clinic', or 'referral'.",
    )


class TriageResponse(BaseModel):
    """Structured clinical triage response."""
    triage_level: str = Field(description="EMERGENCY, URGENT, or ROUTINE")
    assessment: str = Field(description="Full clinical assessment from ClinIQ")
    model: str = Field(description="Model used for inference")
    timestamp: str = Field(description="ISO timestamp of the assessment")
    disclaimer: str = Field(
        default=(
            "⚠️ DISCLAIMER: ClinIQ is a decision-support tool only. "
            "All assessments require validation by a qualified healthcare professional. "
            "This is not a substitute for clinical judgment."
        ),
    )
    processing_time_ms: float = Field(description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    ollama_connected: bool
    timestamp: str


# ===== FastAPI App =====

app = FastAPI(
    title="ClinIQ — Clinical Triage API",
    description=(
        "Offline-first AI clinical triage assistant for community health workers. "
        "Powered by a distilled Qwen2.5-0.5B model running locally via Ollama."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and Ollama are healthy."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            ollama_ok = resp.status_code == 200
    except httpx.RequestError:
        pass

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        model=MODEL_NAME,
        ollama_connected=ollama_ok,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/triage", response_model=TriageResponse, tags=["Clinical"])
async def triage_patient(request: TriageRequest):
    """
    Assess a patient and provide triage classification.
    
    Sends the patient description to the locally-running ClinIQ model
    via Ollama and returns a structured clinical assessment.
    """
    start_time = datetime.utcnow()

    # Build prompt
    prompt = request.patient_description
    if request.urgency_context:
        prompt += f"\n\nContext: This patient is presenting at a {request.urgency_context} level facility."

    # Query Ollama
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.85,
                        "num_predict": 1024,
                    },
                },
            )
            response.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is not reachable. Ensure it is running at {OLLAMA_HOST}. Error: {e}",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {e.response.text}",
        )

    result = response.json()
    assessment = result.get("response", "No response generated.")

    # Parse triage level from response
    triage_level = "ROUTINE"  # default
    assessment_upper = assessment.upper()
    if "EMERGENCY" in assessment_upper or "RED" in assessment_upper:
        triage_level = "EMERGENCY"
    elif "URGENT" in assessment_upper or "YELLOW" in assessment_upper:
        triage_level = "URGENT"

    # Calculate processing time
    end_time = datetime.utcnow()
    processing_ms = (end_time - start_time).total_seconds() * 1000

    return TriageResponse(
        triage_level=triage_level,
        assessment=assessment,
        model=MODEL_NAME,
        timestamp=start_time.isoformat(),
        processing_time_ms=round(processing_ms, 2),
    )


@app.get("/", tags=["System"])
async def root():
    """API root — returns basic info."""
    return {
        "name": "ClinIQ",
        "version": "1.0.0",
        "description": "Offline Clinical Triage Assistant",
        "docs": "/docs",
        "health": "/health",
        "triage": "POST /triage",
    }


if __name__ == "__main__":
    print("=" * 50)
    print("ClinIQ — Clinical Triage API Server")
    print(f"Ollama:  {OLLAMA_HOST}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Server:  http://{SERVE_HOST}:{SERVE_PORT}")
    print(f"Docs:    http://{SERVE_HOST}:{SERVE_PORT}/docs")
    print("=" * 50)

    uvicorn.run(
        "deploy.serve:app",
        host=SERVE_HOST,
        port=SERVE_PORT,
        reload=False,
    )
