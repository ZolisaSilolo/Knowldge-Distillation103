"""
notify.py — ntfy.sh push notification wrapper for pipeline events.

Sends real-time notifications to your phone/desktop when:
- A training stage starts, completes, or fails
- A checkpoint is uploaded to S3
- The model comparison Lambda picks a winner
"""

import requests
from utils.config import get_env


def send_notification(
    title: str,
    message: str,
    priority: str = "default",
    tags: list[str] = None,
) -> bool:
    """
    Send a push notification via ntfy.sh.
    
    Args:
        title: Notification title (e.g., "✅ Stage C Complete").
        message: Notification body text.
        priority: One of "min", "low", "default", "high", "urgent".
        tags: Optional emoji tags (e.g., ["white_check_mark", "robot"]).
        
    Returns:
        bool: True if notification was sent successfully.
    """
    try:
        server = get_env("NTFY_SERVER", "https://ntfy.sh")
        topic = get_env("NTFY_TOPIC", "cliniq-pipeline")
    except EnvironmentError:
        # Notifications are optional — don't crash the pipeline
        return False

    url = f"{server}/{topic}"
    headers = {
        "Title": title,
        "Priority": priority,
    }
    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        response = requests.post(url, data=message, headers=headers, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        # Notification failure should never crash training
        return False


def notify_stage_start(stage_name: str, config: dict = None):
    """Send notification that a training stage has started."""
    msg = f"Training stage {stage_name} started."
    if config:
        msg += f"\nEpochs: {config.get('epochs', '?')}, LR: {config.get('learning_rate', '?')}"
    send_notification(
        title=f"🚀 {stage_name} Started",
        message=msg,
        tags=["rocket"],
    )


def notify_stage_complete(stage_name: str, metrics: dict = None):
    """Send notification that a training stage has completed."""
    msg = f"Training stage {stage_name} completed successfully."
    if metrics:
        msg += f"\nPerplexity: {metrics.get('perplexity', '?'):.4f}"
        msg += f"\nROUGE-L F1: {metrics.get('rouge_l_f1', '?'):.4f}"
    send_notification(
        title=f"✅ {stage_name} Complete",
        message=msg,
        priority="high",
        tags=["white_check_mark"],
    )


def notify_stage_error(stage_name: str, error: str):
    """Send notification that a training stage has failed."""
    send_notification(
        title=f"❌ {stage_name} Failed",
        message=f"Error: {error}",
        priority="urgent",
        tags=["x"],
    )
