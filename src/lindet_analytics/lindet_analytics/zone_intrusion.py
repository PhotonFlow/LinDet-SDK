# Copyright 2024 LinDet Team. Apache-2.0 license.
"""Zone intrusion detection module.

Checks whether tracked object centers fall inside configured polygon zones.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> bool:
    """Ray-casting algorithm for point-in-polygon test.

    Works with normalized coordinates [0, 1].
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


class ZoneIntrusion:
    """Detects when tracked objects enter configured polygon zones.

    Zones are defined as lists of (x, y) vertices in normalized [0, 1] coords.
    """

    def __init__(self, zones: dict):
        """Init with zones dict: {zone_name: [(x1,y1), (x2,y2), ...]}."""
        self.zones = zones
        self._inside: dict = {}  # track_id → set of zone names currently inside

    def check(
        self, track_id: int, cx: float, cy: float
    ) -> List[dict]:
        """Check if a tracked object center is inside any zone.

        Returns list of event dicts for newly entered zones.
        """
        events = []
        prev = self._inside.get(track_id, set())
        curr = set()

        for name, polygon in self.zones.items():
            if point_in_polygon((cx, cy), polygon):
                curr.add(name)
                if name not in prev:
                    events.append({
                        "type": "zone_intrusion",
                        "zone": name,
                        "action": "entered",
                        "position": [cx, cy],
                    })
            elif name in prev:
                events.append({
                    "type": "zone_intrusion",
                    "zone": name,
                    "action": "exited",
                    "position": [cx, cy],
                })

        self._inside[track_id] = curr
        return events
