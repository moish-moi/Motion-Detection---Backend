from typing import Dict, Any


def push_alert(decision: Dict[str, Any]) -> None:
    """
    Push an alert to the frontend / notification system.

    This function is intentionally simple for now.
    It represents an automatic backend-triggered alert
    (NO button, NO manual action).

    Later implementations may include:
    - WebSocket push
    - HTTP POST to frontend
    - Message queue (Kafka / SQS / SNS)
    - Logging / persistence
    """

    event_id = decision.get("event_id", "UNKNOWN")
    label = decision.get("label", "UNKNOWN")
    reason = decision.get("reason", "")

    # --- DEMO / MOCK IMPLEMENTATION ---
    print(
        f"🚨 ALERT PUSHED | event_id={event_id} | label={label} | reason={reason}"
    )
