# Copyright 2024 LinDet Team. Apache-2.0 license.
"""Virtual line crossing counter module.

Counts objects crossing a configured line segment, with direction detection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def cross_product_2d(
    o: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class LineCounter:
    """Counts objects crossing a virtual line segment.

    The line is defined by two endpoints in normalized [0, 1] coordinates.
    A crossing is detected when the tracked object center moves from one
    side of the line to the other between consecutive frames.

    Direction is determined by the sign of the cross product:
      positive → 'in', negative → 'out'
    """

    def __init__(
        self,
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        name: str = "line_0",
    ):
        self.p1 = line_start
        self.p2 = line_end
        self.name = name
        self.count_in = 0
        self.count_out = 0
        self._prev_side: Dict[int, float] = {}  # track_id → side

    def check(
        self, track_id: int, cx: float, cy: float
    ) -> List[dict]:
        """Check if a tracked object crossed the line.

        Returns list of crossing event dicts (usually 0 or 1).
        """
        side = cross_product_2d(self.p1, self.p2, (cx, cy))
        events = []

        prev = self._prev_side.get(track_id)
        if prev is not None and prev * side < 0:
            # Crossed!
            if side > 0:
                self.count_in += 1
                direction = "in"
            else:
                self.count_out += 1
                direction = "out"

            events.append({
                "type": "line_crossing",
                "line": self.name,
                "direction": direction,
                "count_in": self.count_in,
                "count_out": self.count_out,
                "position": [cx, cy],
            })

        self._prev_side[track_id] = side
        return events
