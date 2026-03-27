"""
compare_models/handler.py — Lambda function to compare teacher vs student metrics
and push the winning LoRA adapter to HuggingFace Hub.

Triggered by: EventBridge rule on S3 PutObject in metrics/ prefix.

CONSTRAINT: Lambda has 15-min timeout and limited /tmp storage (512MB-10GB).
We push ONLY the LoRA adapter weights (~100MB), NOT the full merged model.
"""

import json
import os
import tempfile

import boto3


def handler(event, context):
    """
    AWS Lambda handler for model comparison.
    
    1. Fetch teacher and student eval metrics from S3
    2. Compare on key metrics (eval_loss, perplexity)
    3. Push winning LoRA adapter to HuggingFace Hub
    4. Send notification via ntfy.sh
    """
    print(f"🔄 compare_models Lambda triggered")
    print(f"Event: {json.dumps(event, indent=2)}")

    # ===== Configuration =====
    bucket = os.environ.get("S3_BUCKET_NAME", "cliniq-distillation")
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_REPO_ID", "cliniq/cliniq-0.5b-lora")
    ntfy_topic = os.environ.get("NTFY_TOPIC", "cliniq-pipeline")

    s3 = boto3.client("s3")

    # ===== 1. Fetch Metrics =====
    try:
        teacher_metrics = _fetch_metrics(s3, bucket, "metrics/stage_c/eval_metrics.json")
        student_sft_metrics = _fetch_metrics(s3, bucket, "metrics/stage_b/eval_metrics.json")
        student_distilled_metrics = _fetch_metrics(s3, bucket, "metrics/stage_a/eval_metrics.json")
    except Exception as e:
        print(f"❌ Failed to fetch metrics: {e}")
        _notify(ntfy_topic, "❌ Model Comparison Failed", f"Could not fetch metrics: {e}")
        return {"statusCode": 500, "body": f"Metrics fetch failed: {e}"}

    # ===== 2. Compare Models =====
    print("\n📊 Model Comparison Report:")
    print(f"  Teacher (Stage C) — Loss: {teacher_metrics.get('eval_loss', 'N/A')}")
    print(f"  Student SFT (B)   — Loss: {student_sft_metrics.get('eval_loss', 'N/A')}")
    print(f"  Student KD (A)    — Loss: {student_distilled_metrics.get('eval_loss', 'N/A')}")

    # Pick the best student model (distilled should be better than SFT-only)
    winner = "stage_a"
    winner_metrics = student_distilled_metrics
    winner_loss = student_distilled_metrics.get("eval_loss", float("inf"))

    sft_loss = student_sft_metrics.get("eval_loss", float("inf"))
    if sft_loss < winner_loss:
        winner = "stage_b"
        winner_metrics = student_sft_metrics
        winner_loss = sft_loss
        print("⚠️ SFT student outperformed distilled student — using Stage B adapter")
    else:
        print("✅ Distilled student (Stage A) is the winner")

    comparison_report = {
        "winner": winner,
        "teacher_loss": teacher_metrics.get("eval_loss"),
        "student_sft_loss": sft_loss,
        "student_distilled_loss": student_distilled_metrics.get("eval_loss"),
        "winner_loss": winner_loss,
    }

    # Save comparison report to S3
    s3.put_object(
        Bucket=bucket,
        Key="metrics/comparison_report.json",
        Body=json.dumps(comparison_report, indent=2),
        ContentType="application/json",
    )

    # ===== 3. Push LoRA Adapter to HuggingFace Hub =====
    if hf_token:
        try:
            _push_adapter_to_hf(s3, bucket, winner, hf_token, hf_repo)
            print(f"✅ LoRA adapter pushed to HuggingFace: {hf_repo}")
        except Exception as e:
            print(f"⚠️ HF push failed: {e}")
            _notify(ntfy_topic, "⚠️ HF Push Failed", f"Adapter push error: {e}")
    else:
        print("⚠️ HF_TOKEN not set — skipping HuggingFace push")

    # ===== 4. Send Notification =====
    summary = (
        f"Winner: {winner}\n"
        f"Teacher Loss: {teacher_metrics.get('eval_loss', 'N/A')}\n"
        f"Winner Loss: {winner_loss}\n"
        f"Improvement: {((teacher_metrics.get('eval_loss', 0) - winner_loss) / max(teacher_metrics.get('eval_loss', 1), 0.001)) * 100:.1f}%"
    )
    _notify(ntfy_topic, "🏆 Model Comparison Complete", summary)

    return {
        "statusCode": 200,
        "body": json.dumps(comparison_report),
    }


def _fetch_metrics(s3_client, bucket: str, key: str) -> dict:
    """Fetch a JSON metrics file from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def _push_adapter_to_hf(s3_client, bucket: str, stage: str, hf_token: str, repo_id: str):
    """
    Download the LoRA adapter from S3 and push to HuggingFace Hub.
    
    IMPORTANT: Only pushes the LoRA adapter (~100MB), NOT the full merged model.
    This keeps us within Lambda's /tmp storage and execution time limits.
    """
    import tarfile
    from huggingface_hub import HfApi

    s3_key = f"checkpoints/{stage}/{stage}_checkpoint.tar.gz"

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = os.path.join(tmp_dir, "checkpoint.tar.gz")

        # Download adapter archive from S3
        print(f"📥 Downloading adapter from s3://{bucket}/{s3_key}")
        s3_client.download_file(bucket, s3_key, archive_path)

        # Extract (LoRA adapter is small enough for /tmp)
        print("📦 Extracting adapter...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmp_dir)

        # Find the adapter directory
        extracted_dirs = [
            d for d in os.listdir(tmp_dir)
            if os.path.isdir(os.path.join(tmp_dir, d))
        ]
        if not extracted_dirs:
            raise FileNotFoundError("No adapter directory found in archive")

        adapter_dir = os.path.join(tmp_dir, extracted_dirs[0])

        # Push to HuggingFace Hub (LoRA adapter only)
        print(f"🚀 Pushing LoRA adapter to {repo_id}")
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"ClinIQ: Upload {stage} LoRA adapter (auto-deployed by Lambda)",
        )


def _notify(topic: str, title: str, message: str):
    """Send a notification via ntfy.sh."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "high"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Notification failure should never crash the Lambda
