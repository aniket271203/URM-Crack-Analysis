import numpy as np
from typing import List, Tuple, Dict, Optional

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Format: [x, y, w, h]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

def get_bbox_center(bbox: List[int]) -> Tuple[int, int]:
    """Returns (cx, cy) of a bbox [x, y, w, h]."""
    return (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

def is_above(box_a: List[int], box_b: List[int]) -> bool:
    """Returns True if box_a is physically above box_b."""
    # Check if bottom of A is above top of B (with some tolerance?)
    # y increases downwards
    # bottom_a = y_a + h_a
    # top_b = y_b
    return (box_a[1] + box_a[3]) <= box_b[1]

def is_below(box_a: List[int], box_b: List[int]) -> bool:
    """Returns True if box_a is physically below box_b."""
    return box_a[1] >= (box_b[1] + box_b[3])

def is_left_of(box_a: List[int], box_b: List[int]) -> bool:
    """Returns True if box_a is to the left of box_b."""
    return (box_a[0] + box_a[2]) <= box_b[0]

def is_right_of(box_a: List[int], box_b: List[int]) -> bool:
    """Returns True if box_a is to the right of box_b."""
    return box_a[0] >= (box_b[0] + box_b[2])

def rectangles_intersect(r1: List[int], r2: List[int]) -> bool:
    """
    Returns True if two rectangles intersect.
    Rect: [x, y, w, h]
    """
    # x overlap
    if (r1[0] >= r2[0] + r2[2]) or (r2[0] >= r1[0] + r1[2]):
        return False
    # y overlap
    if (r1[1] >= r2[1] + r2[3]) or (r2[1] >= r1[1] + r1[3]):
        return False
    return True
