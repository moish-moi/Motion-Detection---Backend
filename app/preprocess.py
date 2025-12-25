from typing import Dict, Any, List
from app.models import Event, Box
from app.geometry import clip_box, box_area


def preprocess_event(
    event: Event,
    heat_mask,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize and clean an incoming Event from the media stream.

    Responsibilities:
    - Clip boxes to frame boundaries
    - Remove invalid / tiny boxes
    - Return a clean, stable structure for downstream logic

    Returns a dict (event_clean) that is SAFE to use everywhere.
    """

    # Frame size is derived from heat mask (single source of truth)
    height, width = heat_mask.shape

    min_box_area = int(config.get("min_box_area", 0))

    raw_boxes: List[Box] = event.boxes or []

    # Handle empty event early
    if len(raw_boxes) == 0:
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "boxes": [],
            "raw_num_boxes": 0,
            "kept_num_boxes": 0,
            "min_box_area": min_box_area,
        }

    cleaned_boxes: List[Box] = []

    for box in raw_boxes:
        # 1) Clip to frame
        clipped = clip_box(box, width, height)

        # 2) Filter tiny / invalid boxes
        if box_area(clipped) < min_box_area:
            continue

        cleaned_boxes.append(clipped)

    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp,
        "boxes": cleaned_boxes,
        "raw_num_boxes": len(raw_boxes),
        "kept_num_boxes": len(cleaned_boxes),
        "min_box_area": min_box_area,
    }
