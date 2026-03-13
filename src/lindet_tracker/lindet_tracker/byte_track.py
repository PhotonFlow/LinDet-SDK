# Copyright 2024 LinDet Team. Apache-2.0 license.
"""ByteTrack multi-object tracker — pure Python implementation.

ByteTrack (ECCV 2022) uses a two-stage association strategy:
  1. High-confidence detections matched to existing tracks via IoU
  2. Remaining (low-confidence) detections matched to unmatched tracks

This avoids the need for a separate ReID model, making it 2-3x faster
than DeepSORT on Jetson-class hardware.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    # Fallback for environments without scipy
    linear_sum_assignment = None  # type: ignore


# ─── Kalman Filter (constant-velocity model) ─────────────────────────────────

class KalmanFilter:
    """Lightweight Kalman filter for 2D bbox tracking.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement:  [cx, cy, w, h]
    """

    def __init__(self):
        # State transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        self.F[:4, 4:] = np.eye(4, dtype=np.float32)

        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float32)

        # Process noise
        self.Q = np.eye(8, dtype=np.float32) * 1e-2
        self.Q[4:, 4:] *= 10.0

        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1e-1

    def initiate(self, measurement: np.ndarray):
        """Initialize state from first measurement."""
        x = np.zeros(8, dtype=np.float32)
        x[:4] = measurement
        P = np.eye(8, dtype=np.float32) * 10.0
        P[4:, 4:] *= 100.0
        return x, P

    def predict(self, x: np.ndarray, P: np.ndarray):
        """Predict next state."""
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray):
        """Update state with measurement."""
        y = z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return x, P


# ─── Track ────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    """Single tracked object."""
    track_id: int
    class_id: int
    state: np.ndarray          # Kalman state [cx, cy, w, h, vx, vy, vw, vh]
    covariance: np.ndarray     # 8x8 covariance
    age: int = 0               # frames since first detection
    hits: int = 1              # total successful matches
    time_since_update: int = 0
    confidence: float = 0.0

    @property
    def bbox(self) -> np.ndarray:
        """Return [cx, cy, w, h]."""
        return self.state[:4].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Return [vx, vy] in normalized coords per frame."""
        return self.state[4:6].copy()


# ─── ByteTrack ────────────────────────────────────────────────────────────────

class ByteTrack:
    """ByteTrack multi-object tracker.

    Args:
        high_thresh: Confidence threshold for high-quality detections.
        low_thresh:  Confidence threshold for low-quality detections.
        match_thresh: IoU threshold for matching.
        max_age:     Maximum frames to keep a track without updates.
        min_hits:    Minimum hits before a track is considered confirmed.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(
        self,
        detections: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
    ) -> List[Track]:
        """Run one tracking step.

        Args:
            detections: (N, 4) array of [cx, cy, w, h] in normalized coords.
            class_ids: (N,) integer class IDs.
            confidences: (N,) confidence scores.

        Returns:
            List of confirmed Track objects with updated states.
        """
        # ── Predict all tracks ─────────────────────────────────────────────
        for track in self.tracks:
            track.state, track.covariance = self.kf.predict(
                track.state, track.covariance)

        # ── Split detections into high/low confidence ──────────────────────
        high_mask = confidences >= self.high_thresh
        low_mask = (confidences >= self.low_thresh) & ~high_mask

        high_dets = detections[high_mask]
        high_cls = class_ids[high_mask]
        high_conf = confidences[high_mask]

        low_dets = detections[low_mask]
        low_cls = class_ids[low_mask]
        low_conf = confidences[low_mask]

        # ── STAGE 1: Match high-confidence detections to all tracks ────────
        track_bboxes = np.array([t.bbox for t in self.tracks]) if self.tracks else np.empty((0, 4))
        iou_matrix = self._iou_batch(high_dets, track_bboxes)

        matched, unmatched_dets, unmatched_tracks = self._linear_assignment(
            iou_matrix, self.match_thresh)

        # Update matched tracks
        for det_idx, trk_idx in matched:
            track = self.tracks[trk_idx]
            z = high_dets[det_idx]
            track.state, track.covariance = self.kf.update(
                track.state, track.covariance, z)
            track.hits += 1
            track.time_since_update = 0
            track.confidence = float(high_conf[det_idx])
            track.class_id = int(high_cls[det_idx])

        # ── STAGE 2: Match low-confidence detections to remaining tracks ───
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        if len(low_dets) > 0 and len(remaining_tracks) > 0:
            rem_bboxes = np.array([t.bbox for t in remaining_tracks])
            iou_matrix2 = self._iou_batch(low_dets, rem_bboxes)

            matched2, unmatched_dets2, still_unmatched = self._linear_assignment(
                iou_matrix2, self.match_thresh)

            for det_idx, trk_idx in matched2:
                track = remaining_tracks[trk_idx]
                z = low_dets[det_idx]
                track.state, track.covariance = self.kf.update(
                    track.state, track.covariance, z)
                track.hits += 1
                track.time_since_update = 0
                track.confidence = float(low_conf[det_idx])

            # Remove matched tracks from unmatched set
            matched_track_indices = {unmatched_tracks[trk_idx] for _, trk_idx in matched2}
            unmatched_tracks = [i for i in unmatched_tracks if i not in matched_track_indices]

        # ── Create new tracks for unmatched high-confidence detections ─────
        for det_idx in unmatched_dets:
            z = high_dets[det_idx]
            x, P = self.kf.initiate(z)
            self.tracks.append(Track(
                track_id=self._next_id,
                class_id=int(high_cls[det_idx]),
                state=x,
                covariance=P,
                confidence=float(high_conf[det_idx]),
            ))
            self._next_id += 1

        # ── Age all tracks, remove old ones ────────────────────────────────
        for track in self.tracks:
            track.age += 1
            if track.time_since_update > 0:
                track.time_since_update += 1

        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # Increment time_since_update for unmatched tracks
        for idx in unmatched_tracks:
            if idx < len(self.tracks):
                self.tracks[idx].time_since_update += 1

        # ── Return confirmed tracks ───────────────────────────────────────
        return [t for t in self.tracks if t.hits >= self.min_hits]

    @staticmethod
    def _iou_batch(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between detections and tracks.

        Both inputs are (N, 4) arrays of [cx, cy, w, h].
        Returns (N_dets, N_trks) IoU matrix.
        """
        if len(dets) == 0 or len(trks) == 0:
            return np.empty((len(dets), len(trks)))

        # Convert to x1, y1, x2, y2
        d = np.column_stack([
            dets[:, 0] - dets[:, 2] / 2,
            dets[:, 1] - dets[:, 3] / 2,
            dets[:, 0] + dets[:, 2] / 2,
            dets[:, 1] + dets[:, 3] / 2,
        ])
        t = np.column_stack([
            trks[:, 0] - trks[:, 2] / 2,
            trks[:, 1] - trks[:, 3] / 2,
            trks[:, 0] + trks[:, 2] / 2,
            trks[:, 1] + trks[:, 3] / 2,
        ])

        xx1 = np.maximum(d[:, None, 0], t[None, :, 0])
        yy1 = np.maximum(d[:, None, 1], t[None, :, 1])
        xx2 = np.minimum(d[:, None, 2], t[None, :, 2])
        yy2 = np.minimum(d[:, None, 3], t[None, :, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_d = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
        area_t = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
        union = area_d[:, None] + area_t[None, :] - inter

        return inter / (union + 1e-6)

    @staticmethod
    def _linear_assignment(
        cost_matrix: np.ndarray, thresh: float
    ):
        """Hungarian algorithm matching with threshold."""
        if cost_matrix.size == 0:
            return (
                [],
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])),
            )

        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        else:
            # Greedy fallback
            row_ind, col_ind = [], []
            used_cols = set()
            for r in range(cost_matrix.shape[0]):
                best_c = -1
                best_v = -1
                for c in range(cost_matrix.shape[1]):
                    if c not in used_cols and cost_matrix[r, c] > best_v:
                        best_v = cost_matrix[r, c]
                        best_c = c
                if best_c >= 0:
                    row_ind.append(r)
                    col_ind.append(best_c)
                    used_cols.add(best_c)
            row_ind = np.array(row_ind)
            col_ind = np.array(col_ind)

        matched = []
        unmatched_dets = list(range(cost_matrix.shape[0]))
        unmatched_trks = list(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= thresh:
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks
