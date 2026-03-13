# Copyright 2024 LinDet Team. Apache-2.0 license.
"""Speed estimation module.

Estimates object speed from track velocity using optional homography
calibration for real-world unit conversion.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


class SpeedEstimator:
    """Estimates speed of tracked objects.

    If a homography matrix is provided, velocity is converted from pixel
    coordinates to real-world units (e.g., meters/second).  Otherwise,
    speed is reported in normalized-coordinate units per frame.

    Args:
        fps: Camera framerate for time conversion.
        homography: Optional 3x3 homography matrix (pixel → world).
        speed_alert_threshold: Speed above which an alert is fired.
    """

    def __init__(
        self,
        fps: float = 30.0,
        homography: Optional[np.ndarray] = None,
        speed_alert_threshold: float = float("inf"),
    ):
        self.fps = fps
        self.H = homography  # 3x3, maps image coords → ground plane
        self.threshold = speed_alert_threshold

    def check(
        self,
        track_id: int,
        vx: float,
        vy: float,
        cx: float,
        cy: float,
    ) -> List[dict]:
        """Estimate speed and return alert events if threshold exceeded.

        Args:
            track_id: Track identifier.
            vx, vy: Velocity in normalized coords per frame.
            cx, cy: Current center position (for homography transform).

        Returns:
            List of event dicts (usually 0 or 1).
        """
        if self.H is not None:
            # Transform velocity through homography
            # Approximate: transform (cx, cy) and (cx+vx, cy+vy), take diff
            p1 = self._warp_point(cx, cy)
            p2 = self._warp_point(cx + vx, cy + vy)
            world_vx = (p2[0] - p1[0]) * self.fps
            world_vy = (p2[1] - p1[1]) * self.fps
            speed = float(np.sqrt(world_vx ** 2 + world_vy ** 2))
            unit = "m/s"
        else:
            speed = float(np.sqrt(vx ** 2 + vy ** 2)) * self.fps
            unit = "norm/s"

        events = []
        if speed > self.threshold:
            events.append({
                "type": "speed_alert",
                "speed": round(speed, 2),
                "unit": unit,
                "position": [cx, cy],
            })

        return events

    def _warp_point(self, x: float, y: float):
        """Apply homography to a single point."""
        p = np.array([x, y, 1.0])
        wp = self.H @ p
        return wp[:2] / (wp[2] + 1e-9)
