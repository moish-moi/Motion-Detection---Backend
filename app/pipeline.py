from typing import Dict, Any, Optional

from app.models import Event
from app.preprocess import preprocess_event
from app.overlap import compute_event_overlap_with_mask
from app.decision import decide_ignore_or_forward
from app.alerts import push_alert


def filter_event(
    event: Event,
    heat_mask,
    config: Dict[str, Any],
    time_score: Optional[float] = None,
    auto_push: bool = True,
) -> Dict[str, Any]:
    """
    Main ONLINE pipeline.
    Called for every event coming from the media stream.

    Steps:
    1) Preprocess event
    2) Compute overlap vs heat mask
    3) Decide IGNORE / FORWARD
    4) Push alert automatically if FORWARD

    Returns:
        decision dict
    """

    # 1) Preprocess
    event_clean = preprocess_event(event, heat_mask, config)

    # 2) Overlap computation
    overlap_stats = compute_event_overlap_with_mask(
        event_clean, heat_mask, config
    )

    # 3) Decision
    decision = decide_ignore_or_forward(
        event_clean,
        overlap_stats,
        config,
        time_score=time_score,
    )

    # 4) Automatic push (NO BUTTON)
    if auto_push and decision["label"] == "FORWARD":
        push_alert(decision)

    return decision
