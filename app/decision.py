from typing import Dict, Any, Optional, List

from app.models import Box


def decide_ignore_or_forward(
    event_clean: Dict[str, Any],
    overlap_stats: Dict[str, Any],
    config: Dict[str, Any],
    time_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Decide whether to IGNORE or FORWARD an event based on overlap statistics
    and safety rules.

    This is the ONLY place where business logic lives.
    """

    boxes: List[Box] = event_clean["boxes"]

    T_mean = float(config.get("T_mean", 0.60))
    T_ratio = float(config.get("T_ratio", 0.70))

    # Safety configuration
    enable_safety = bool(config.get("enable_min_overlap_safety", True))
    safety_only_when_multiple_boxes = bool(
        config.get("safety_only_when_multiple_boxes", True)
    )
    min_overlap_any_box_forward = float(
        config.get("min_overlap_any_box_forward", 0.10)
    )

    # Case 1: no boxes at all
    if len(boxes) == 0:
        return {
            "label": "IGNORE",
            "reason": "no boxes in event (or all filtered)",
            "metrics": {
                "raw_num_boxes": event_clean.get("raw_num_boxes", 0),
                "kept_num_boxes": event_clean.get("kept_num_boxes", 0),
            },
            "event_id": event_clean["event_id"],
            "timestamp": event_clean["timestamp"].isoformat(),
            "boxes": boxes,
        }

    mean_overlap = overlap_stats["mean_overlap"]
    max_overlap = overlap_stats["max_overlap"]
    min_overlap = overlap_stats["min_overlap"]
    hot_boxes_ratio = overlap_stats["hot_boxes_ratio"]

    # Case 2: SAFETY — mixed boxes, at least one clearly anomalous
    if enable_safety:
        if (not safety_only_when_multiple_boxes) or (len(boxes) >= 2):
            if min_overlap < min_overlap_any_box_forward:
                return {
                    "label": "FORWARD",
                    "reason": (
                        f"min_overlap={min_overlap:.2f} "
                        f"< {min_overlap_any_box_forward:.2f} (safety)"
                    ),
                    "metrics": {
                        "mean_overlap": mean_overlap,
                        "max_overlap": max_overlap,
                        "min_overlap": min_overlap,
                        "hot_boxes_ratio": hot_boxes_ratio,
                        "T_mean": T_mean,
                        "T_ratio": T_ratio,
                        "time_score": time_score,
                    },
                    "event_id": event_clean["event_id"],
                    "timestamp": event_clean["timestamp"].isoformat(),
                    "boxes": boxes,
                }

    # Case 3: Base decision rule
    ignore = (mean_overlap >= T_mean) or (hot_boxes_ratio >= T_ratio)

    label = "IGNORE" if ignore else "FORWARD"
    reason = (
        f"mean_overlap={mean_overlap:.2f}, "
        f"max_overlap={max_overlap:.2f}, "
        f"hot_boxes_ratio={hot_boxes_ratio:.2f}"
    )

    return {
        "label": label,
        "reason": reason,
        "metrics": {
            "mean_overlap": mean_overlap,
            "max_overlap": max_overlap,
            "min_overlap": min_overlap,
            "hot_boxes_ratio": hot_boxes_ratio,
            "T_mean": T_mean,
            "T_ratio": T_ratio,
            "time_score": time_score,
        },
        "event_id": event_clean["event_id"],
        "timestamp": event_clean["timestamp"].isoformat(),
        "boxes": boxes,
    }
