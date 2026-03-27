"""
notify/handler.py — Lightweight Lambda for ntfy.sh notifications.

Triggered by EventBridge rules on S3 events to send real-time
pipeline notifications to your phone/desktop.
"""

import json
import os
import urllib.request


def handler(event, context):
    """
    AWS Lambda handler for pipeline notifications.
    
    Receives S3 event metadata via EventBridge and sends
    a formatted notification through ntfy.sh.
    """
    topic = os.environ.get("NTFY_TOPIC", "cliniq-pipeline")
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

    # Parse event
    s3_key = "unknown"
    bucket = "unknown"

    try:
        detail = event.get("detail", {})
        s3_key = detail.get("object", {}).get("key", "unknown")
        bucket = detail.get("bucket", {}).get("name", "unknown")
    except (KeyError, AttributeError):
        pass

    # Determine notification content based on S3 key
    if "stage_c" in s3_key:
        title = "📦 Stage C: Teacher SFT"
        stage = "Teacher SFT (Qwen2.5-1.5B)"
    elif "stage_b" in s3_key:
        title = "📦 Stage B: Student SFT"
        stage = "Student SFT (Qwen2.5-0.5B)"
    elif "stage_a" in s3_key:
        title = "📦 Stage A: KL Distillation"
        stage = "KL Logit Distillation"
    elif "comparison" in s3_key:
        title = "🏆 Model Comparison"
        stage = "Final comparison report"
    else:
        title = "📌 Pipeline Event"
        stage = "Unknown stage"

    if "checkpoint" in s3_key:
        message = f"{stage} checkpoint uploaded.\nKey: {s3_key}\nBucket: {bucket}"
        priority = "high"
    elif "metrics" in s3_key:
        message = f"{stage} metrics saved.\nKey: {s3_key}"
        priority = "default"
    else:
        message = f"S3 event: {s3_key}"
        priority = "low"

    # Send notification
    try:
        url = f"{server}/{topic}"
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
            },
        )
        response = urllib.request.urlopen(req, timeout=10)
        status = response.status
    except Exception as e:
        status = 0
        print(f"⚠️ Notification failed: {e}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "notification_sent": status == 200,
            "title": title,
            "s3_key": s3_key,
        }),
    }
