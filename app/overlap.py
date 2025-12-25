from typing import Dict, Any, List
import numpy as np

from app.models import Box
from app.geometry import clip_box, box_area


def box_hot_coverage(box: Box, heat_mask: np.ndarray) -> float:
    """
    Computes the ratio of 'hot' pixels (==1) inside a bounding box.
    Returns a value in [0, 1].
    """
    height, width = heat_mask.shape

    # Clip box to frame
    x1, y1, x2, y2 = clip_box(box, width, height)

    area = box_area((x1, y1, x2, y2))
    if area == 0:
        return 0.0

    patch = heat_mask[y1:y2, x1:x2]
    hot_pixels = int(patch.sum())

    return hot_pixels / area


def compute_event_overlap_with_mask(
    event_clean: Dict[str, Any],
    heat_mask: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes overlap statistics of a cleaned event vs the heat mask.

    Returns metrics only — NO decisions here.
    """

    boxes: List[Box] = event_clean["boxes"]
    per_box_threshold = float(config.get("per_box_threshold", 0.60))

    # No boxes => no overlap
    if not boxes:
        return {
            "per_box_overlap": [],
            "mean_overlap": 0.0,
            "max_overlap": 0.0,
            "min_overlap": 0.0,
            "hot_boxes_ratio": 0.0,
            "per_box_threshold": per_box_threshold,
        }

    overlaps = [box_hot_coverage(box, heat_mask) for box in boxes]

    mean_overlap = float(np.mean(overlaps))
    max_overlap = float(np.max(overlaps))
    min_overlap = float(np.min(overlaps))
    hot_boxes_ratio = float(
        np.mean([1.0 if o >= per_box_threshold else 0.0 for o in overlaps])
    )

    return {
        "per_box_overlap": overlaps,
        "mean_overlap": mean_overlap,
        "max_overlap": max_overlap,
        "min_overlap": min_overlap,
        "hot_boxes_ratio": hot_boxes_ratio,
        "per_box_threshold": per_box_threshold,
    }
