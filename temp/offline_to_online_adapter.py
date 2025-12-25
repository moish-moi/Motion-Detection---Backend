from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime

from app.models import Event, Box

BoxXYWH = Tuple[int, int, int, int]


def xywh_to_xyxy(box: BoxXYWH) -> Box:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def normalize_mask_255_to_01(mask_255: np.ndarray) -> np.ndarray:
    """
    Convert binary mask from {0,255} to {0,1} uint8.
    """
    return (mask_255 > 0).astype(np.uint8)


def offline_event_dict_to_event(event_dict, idx):
    # המרת המלבנים מ-xywh ל-xyxy כפי שהשותף מצפה
    boxes_xyxy = []
    for (x, y, w, h) in event_dict.get("boxes", []):
        boxes_xyxy.append((x, y, x + w, y + h))

    # החזרת אובייקט Event נקי
    return Event(
        event_id=f"offline_{idx}",
        timestamp=event_dict["timestamp"],
        boxes=boxes_xyxy
    )

def offline_events_to_online_events(events_offline):
    return [offline_event_dict_to_event(e, i) for i, e in enumerate(events_offline)]