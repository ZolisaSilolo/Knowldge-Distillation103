"""
test_serve.py — Test the FastAPI clinical triage API.

Uses FastAPI TestClient — no Ollama needed.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from deploy.serve import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "ollama_connected" in data

    def test_health_shows_model_name(self, client):
        """Health response should include model name."""
        response = client.get("/health")
        data = response.json()
        assert data["model"] == "cliniq"


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_info(self, client):
        """Root should return API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ClinIQ"
        assert "triage" in data


class TestTriageEndpoint:
    """Test the /triage endpoint."""

    @patch("deploy.serve.httpx.AsyncClient")
    def test_triage_returns_structured_response(self, mock_httpx, client):
        """Triage response should have all required fields."""
        # Mock Ollama response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "TRIAGE LEVEL: EMERGENCY (Red). Patient needs immediate care."
        }
        mock_response.raise_for_status = lambda: None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.return_value = mock_client

        response = client.post(
            "/triage",
            json={"patient_description": "Patient with severe bleeding and loss of consciousness"},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "triage_level" in data
        assert "assessment" in data
        assert "disclaimer" in data
        assert "processing_time_ms" in data
        assert data["triage_level"] == "EMERGENCY"

    @patch("deploy.serve.httpx.AsyncClient")
    def test_disclaimer_always_present(self, mock_httpx, client):
        """Every response MUST include the clinical disclaimer."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "TRIAGE LEVEL: ROUTINE. Mild symptoms."}
        mock_response.raise_for_status = lambda: None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.return_value = mock_client

        response = client.post(
            "/triage",
            json={"patient_description": "Patient with mild headache for 1 day"},
        )
        data = response.json()
        assert "DISCLAIMER" in data["disclaimer"].upper()
        assert "decision-support" in data["disclaimer"].lower()

    def test_triage_rejects_short_input(self, client):
        """Short patient descriptions should be rejected."""
        response = client.post(
            "/triage",
            json={"patient_description": "fever"},
        )
        assert response.status_code == 422  # Validation error
