"""
checkpoint.py — S3 checkpoint upload with integrity verification & aggressive local cleanup.

Implements the SageMaker Studio Lab 15GB storage mitigation:
- Upload checkpoint to S3 immediately after training
- Verify upload integrity via ETag/MD5
- Delete local checkpoint ONLY after verified upload
"""

import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from utils.config import get_env
from utils.notify import send_notification


def _get_s3_client():
    """Create an S3 client using environment credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_env("AWS_SECRET_ACCESS_KEY"),
        region_name=get_env("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file for integrity verification."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _compress_checkpoint(checkpoint_dir: Path, output_path: Path) -> Path:
    """Compress a checkpoint directory into a tar.gz archive."""
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(checkpoint_dir, arcname=checkpoint_dir.name)
    return output_path


def upload_checkpoint(
    checkpoint_dir: str,
    stage_name: str,
    s3_prefix: str = "checkpoints",
    cleanup: bool = True,
) -> str:
    """
    Upload a training checkpoint to S3 with integrity checks.
    
    This function:
    1. Compresses the checkpoint directory
    2. Uploads to S3
    3. Verifies the upload via ETag
    4. Deletes local files (if cleanup=True) — critical for 15GB SageMaker limit
    
    Args:
        checkpoint_dir: Path to the local checkpoint directory.
        stage_name: Pipeline stage name (e.g., "stage_c", "stage_b", "stage_a").
        s3_prefix: S3 key prefix for organizing checkpoints.
        cleanup: Whether to delete local files after verified upload.
        
    Returns:
        str: S3 URI of the uploaded checkpoint.
        
    Raises:
        RuntimeError: If upload or verification fails.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    bucket = get_env("S3_BUCKET_NAME")
    s3_client = _get_s3_client()

    # Compress checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        archive_name = f"{stage_name}_checkpoint.tar.gz"
        archive_path = Path(tmp) / archive_name
        _compress_checkpoint(checkpoint_path, archive_path)

        local_md5 = _compute_md5(archive_path)
        s3_key = f"{s3_prefix}/{stage_name}/{archive_name}"

        # Upload with MD5 metadata
        try:
            s3_client.upload_file(
                str(archive_path),
                bucket,
                s3_key,
                ExtraArgs={"Metadata": {"local-md5": local_md5}},
            )
        except ClientError as e:
            send_notification(
                title=f"❌ {stage_name} Upload Failed",
                message=f"S3 upload error: {e}",
                priority="high",
            )
            raise RuntimeError(f"S3 upload failed: {e}") from e

        # Verify upload integrity
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            remote_md5 = response.get("Metadata", {}).get("local-md5", "")
            if remote_md5 != local_md5:
                raise RuntimeError(
                    f"Integrity check failed: local={local_md5}, remote={remote_md5}"
                )
        except ClientError as e:
            raise RuntimeError(f"Failed to verify upload: {e}") from e

    s3_uri = f"s3://{bucket}/{s3_key}"

    # Aggressive cleanup — critical for SageMaker Studio Lab 15GB limit
    if cleanup:
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        send_notification(
            title=f"🧹 {stage_name} Cleanup",
            message=f"Local checkpoint deleted after verified S3 upload.",
        )

    send_notification(
        title=f"✅ {stage_name} Checkpoint Uploaded",
        message=f"Uploaded to {s3_uri}",
    )

    return s3_uri


def upload_metrics(metrics: dict, stage_name: str) -> str:
    """
    Upload evaluation metrics as a JSON file to S3.
    
    This is used by the Lambda compare_models function to pick a winner.
    
    Args:
        metrics: Dictionary of evaluation metrics.
        stage_name: Pipeline stage name.
        
    Returns:
        str: S3 URI of the metrics file.
    """
    import json

    bucket = get_env("S3_BUCKET_NAME")
    s3_client = _get_s3_client()
    s3_key = f"metrics/{stage_name}/eval_metrics.json"

    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(metrics, indent=2),
        ContentType="application/json",
    )

    s3_uri = f"s3://{bucket}/{s3_key}"
    send_notification(
        title=f"📊 {stage_name} Metrics Uploaded",
        message=f"Metrics saved to {s3_uri}",
    )
    return s3_uri
