# Copyright 2024 LinDet Team. Apache-2.0 license.
"""Drawing utilities for visualization overlay."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# COCO skeleton connections for pose drawing
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
]

# Color palette for different track IDs (BGR)
TRACK_COLORS = [
    (230, 159, 0),    # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # green
    (240, 228, 66),   # yellow
    (0, 114, 178),    # blue
    (213, 94, 0),     # vermillion
    (204, 121, 167),  # pink
    (0, 0, 0),        # black
]

CLASS_COLORS = {
    0: (0, 255, 0),     # person → green
    2: (255, 0, 0),     # car → blue
    3: (255, 165, 0),   # motorcycle → orange
    5: (0, 0, 255),     # bus → red
    7: (128, 0, 128),   # truck → purple
}


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for a detection class."""
    return CLASS_COLORS.get(class_id, (200, 200, 200))


def draw_bbox(
    img: np.ndarray,
    cx: float, cy: float, w: float, h: float,
    color: Tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> None:
    """Draw a bounding box on the image with optional label.

    Coordinates are normalized [0, 1].
    """
    try:
        import cv2
    except ImportError:
        return

    ih, iw = img.shape[:2]
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton(
    img: np.ndarray,
    keypoints: List,
    color: Tuple[int, int, int] = (0, 255, 0),
    kpt_thresh: float = 0.3,
) -> None:
    """Draw pose skeleton on image.

    keypoints: list of Keypoint2D messages with x, y, confidence, id.
    """
    try:
        import cv2
    except ImportError:
        return

    ih, iw = img.shape[:2]
    pts = {}

    for kp in keypoints:
        if kp.confidence >= kpt_thresh:
            px = int(kp.x * iw)
            py = int(kp.y * ih)
            pts[kp.id] = (px, py)
            cv2.circle(img, (px, py), 3, color, -1)

    for i, j in COCO_SKELETON:
        if i in pts and j in pts:
            cv2.line(img, pts[i], pts[j], color, 2, cv2.LINE_AA)


def draw_zone(
    img: np.ndarray,
    polygon: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 200, 200),
    alpha: float = 0.2,
    label: str = "",
) -> None:
    """Draw a semi-transparent polygon zone on the image."""
    try:
        import cv2
    except ImportError:
        return

    ih, iw = img.shape[:2]
    pts = np.array(
        [[int(x * iw), int(y * ih)] for x, y in polygon],
        dtype=np.int32,
    )

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, color, 2)

    if label and len(pts) > 0:
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(img, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_line(
    img: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 0, 255),
    count_in: int = 0,
    count_out: int = 0,
    label: str = "",
) -> None:
    """Draw a counting line on the image with counts."""
    try:
        import cv2
    except ImportError:
        return

    ih, iw = img.shape[:2]
    pt1 = (int(p1[0] * iw), int(p1[1] * ih))
    pt2 = (int(p2[0] * iw), int(p2[1] * ih))

    cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    text = f"{label} In:{count_in} Out:{count_out}"
    cv2.putText(img, text, (mid[0] - 50, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
