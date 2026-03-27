"""
test_dataset.py — Validate data pipeline and tokenization.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_data_file():
    """Create a temporary JSONL file with sample training data."""
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are ClinIQ."},
                {"role": "user", "content": "Patient has fever and cough."},
                {"role": "assistant", "content": "TRIAGE LEVEL: URGENT. Assess for TB."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are ClinIQ."},
                {"role": "user", "content": "Child with diarrhea and sunken eyes."},
                {"role": "assistant", "content": "TRIAGE LEVEL: EMERGENCY. Severe dehydration."},
            ]
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        return f.name


class TestDataPipeline:
    """Test data preparation and loading."""

    def test_jsonl_format(self, sample_data_file):
        """Verify JSONL files have correct structure."""
        with open(sample_data_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                assert "messages" in data, "Each line must have 'messages' field"
                assert len(data["messages"]) >= 2, "Must have at least system + user messages"

    def test_message_roles(self, sample_data_file):
        """Verify messages have valid roles."""
        valid_roles = {"system", "user", "assistant"}
        with open(sample_data_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                for msg in data["messages"]:
                    assert msg["role"] in valid_roles, f"Invalid role: {msg['role']}"
                    assert len(msg["content"]) > 0, "Message content must not be empty"

    def test_clinical_content_present(self, sample_data_file):
        """Verify training data contains clinical content."""
        clinical_keywords = ["triage", "fever", "cough", "emergency", "urgent", "patient"]
        with open(sample_data_file, "r") as f:
            all_text = f.read().lower()
            found = [kw for kw in clinical_keywords if kw in all_text]
            assert len(found) >= 3, f"Expected clinical keywords, found only: {found}"

    def test_system_prompt_exists(self):
        """Verify the system prompt file is present and non-empty."""
        prompt_path = Path(__file__).resolve().parent.parent / "data" / "prompts" / "system_prompt.txt"
        assert prompt_path.exists(), f"System prompt not found at {prompt_path}"
        content = prompt_path.read_text()
        assert len(content) > 100, "System prompt is too short"
        assert "ClinIQ" in content, "System prompt must mention ClinIQ"
        assert "DISCLAIMER" in content.upper(), "System prompt must include disclaimer"


class TestTemplates:
    """Test prompt template formatting."""

    def test_sft_format(self):
        """Verify SFT format produces valid chat messages."""
        from data.prompts.templates import format_sft_example

        messages = format_sft_example("Patient has fever.", "TRIAGE: URGENT")
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "fever" in messages[1]["content"]

    def test_inference_format(self):
        """Verify inference format omits assistant response."""
        from data.prompts.templates import format_inference_prompt

        messages = format_inference_prompt("Patient has cough.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_distillation_format(self):
        """Verify distillation format includes metadata."""
        from data.prompts.templates import format_distillation_example

        result = format_distillation_example(
            "Patient query",
            "Teacher response",
            "/path/to/logits.pt",
        )
        assert "messages" in result
        assert result["teacher_logits_path"] == "/path/to/logits.pt"
