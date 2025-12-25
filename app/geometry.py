from typing import Tuple
from app.models import Box


def clip_box(box: Box, width: int, height: int) -> Box:
    """
    Clips a bounding box to image/frame boundaries.
    Ensures coordinates stay within [0, width/height].
    """
    x1, y1, x2, y2 = box

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    return x1, y1, x2, y2


def box_area(box: Box) -> int:
    """
    Computes area of a bounding box.
    Returns 0 if box is invalid.
    """
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box_a: Box, box_b: Box) -> int:
    """
    Computes intersection area between two bounding boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0

    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def iou(box_a: Box, box_b: Box) -> float:
    """
    Intersection over Union between two boxes.
    Value in [0, 1].
    """
    inter = intersection_area(box_a, box_b)
    if inter == 0:
        return 0.0

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - inter
    if union == 0:
        return 0.0

    return inter / union
