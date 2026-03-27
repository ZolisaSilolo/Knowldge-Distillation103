"""
test_checkpoint.py — Test S3 checkpoint upload and cleanup logic.

Uses moto to mock S3 — no real AWS calls needed.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws


@pytest.fixture
def mock_env():
    """Set required environment variables for testing."""
    env = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "S3_BUCKET_NAME": "test-cliniq-bucket",
        "NTFY_TOPIC": "test-topic",
        "NTFY_SERVER": "https://ntfy.sh",
    }
    with patch.dict(os.environ, env):
        yield env


@pytest.fixture
def mock_s3(mock_env):
    """Create a mock S3 bucket."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-cliniq-bucket")
        yield s3


@pytest.fixture
def sample_checkpoint():
    """Create a temporary checkpoint directory with dummy files."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "test_checkpoint"
        ckpt_dir.mkdir()
        
        # Create dummy adapter files
        (ckpt_dir / "adapter_config.json").write_text('{"r": 16}')
        (ckpt_dir / "adapter_model.bin").write_bytes(b"dummy_weights" * 100)
        (ckpt_dir / "tokenizer_config.json").write_text('{"model_type": "qwen2"}')
        
        yield str(ckpt_dir)


class TestCheckpointUpload:
    """Test S3 checkpoint upload functionality."""

    @mock_aws
    def test_upload_creates_s3_object(self, mock_env, sample_checkpoint):
        """Verify checkpoint is uploaded to S3."""
        # Re-create bucket inside mock context
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-cliniq-bucket")

        with patch("utils.checkpoint.send_notification"):
            from utils.checkpoint import upload_checkpoint
            s3_uri = upload_checkpoint(
                checkpoint_dir=sample_checkpoint,
                stage_name="stage_c",
                cleanup=False,  # Don't delete fixture
            )

        assert s3_uri.startswith("s3://test-cliniq-bucket/")
        assert "stage_c" in s3_uri

    @mock_aws
    def test_cleanup_deletes_local_files(self, mock_env):
        """Verify local checkpoint is deleted after upload."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-cliniq-bucket")

        # Create a temporary checkpoint that SHOULD be deleted
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "cleanup_test"
            ckpt_dir.mkdir()
            (ckpt_dir / "model.bin").write_bytes(b"weights" * 50)

            assert ckpt_dir.exists(), "Checkpoint should exist before upload"

            with patch("utils.checkpoint.send_notification"):
                from utils.checkpoint import upload_checkpoint
                upload_checkpoint(
                    checkpoint_dir=str(ckpt_dir),
                    stage_name="stage_test",
                    cleanup=True,
                )

            assert not ckpt_dir.exists(), "Checkpoint should be DELETED after upload"

    @mock_aws
    def test_metrics_upload(self, mock_env):
        """Verify metrics JSON is uploaded to S3."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-cliniq-bucket")

        metrics = {"eval_loss": 0.45, "perplexity": 12.3}

        with patch("utils.checkpoint.send_notification"):
            from utils.checkpoint import upload_metrics
            s3_uri = upload_metrics(metrics, "stage_c")

        assert "metrics/stage_c/" in s3_uri

        # Verify content
        response = s3.get_object(
            Bucket="test-cliniq-bucket",
            Key="metrics/stage_c/eval_metrics.json",
        )
        stored = json.loads(response["Body"].read())
        assert stored["eval_loss"] == 0.45


class TestCheckpointEdgeCases:
    """Test error handling and edge cases."""

    def test_nonexistent_checkpoint_raises(self, mock_env):
        """Verify FileNotFoundError for missing checkpoint."""
        with patch("utils.checkpoint.send_notification"):
            from utils.checkpoint import upload_checkpoint
            with pytest.raises(FileNotFoundError):
                upload_checkpoint(
                    checkpoint_dir="/nonexistent/path",
                    stage_name="stage_x",
                )
