from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from datetime import datetime


# ---------- BASIC TYPES ----------

# Bounding Box: (x1, y1, x2, y2) in pixel coordinates
Box = Tuple[int, int, int, int]


# ---------- EVENT MODEL ----------

@dataclass
class Event:
    """
    Represents a motion event coming from the media stream.
    This is the INPUT of the online pipeline.
    """
    event_id: str
    timestamp: datetime
    boxes: List[Box]


# ---------- DECISION MODEL ----------

@dataclass
class Decision:
    """
    Represents the output decision of the online pipeline.
    This is what gets pushed to the frontend on FORWARD.
    """
    label: str                 # "IGNORE" or "FORWARD"
    reason: str                # Human-readable explanation
    metrics: Dict[str, Any]    # Debug / explainability metrics
    event_id: str
    timestamp: str             # ISO string
    boxes: List[Box]
