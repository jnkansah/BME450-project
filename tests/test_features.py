"""
Tests for EAR, MAR, and head-pose feature computation.
"""
import math
import types
import pytest
import numpy as np

from src.features import eye_aspect_ratio, mouth_aspect_ratio, head_pose_angles
from src.config import EAR_THRESH, MAR_THRESH, HEAD_PITCH_THRESH


# ──────────────────────────────────────────────────────────────────────────────
# Helpers – build a minimal fake landmark list
# ──────────────────────────────────────────────────────────────────────────────

def _make_landmark(x, y, z=0.0):
    lm = types.SimpleNamespace()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _landmark_list(n=500):
    """Return a list of n landmarks all at (0.5, 0.5)."""
    return [_make_landmark(0.5, 0.5) for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# EAR tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEAR:
    def test_fully_open_eye(self):
        """
        When the vertical distances are equal to horizontal, EAR ≈ 1.
        Layout: p1(0,0.5), p2(0.2,0), p3(0.4,0), p4(0.6,0.5),
                p5(0.4,1), p6(0.2,1)  → roughly circular → EAR near 0.5+
        """
        # Use a specific 6-point arrangement for a wide-open eye
        lm = _landmark_list()
        W, H = 100, 100
        indices = [0, 1, 2, 3, 4, 5]
        # p1=left, p4=right, p2/p3=top, p5/p6=bottom
        lm[0] = _make_landmark(0.0, 0.5)   # p1
        lm[1] = _make_landmark(0.25, 0.0)  # p2 top-left
        lm[2] = _make_landmark(0.75, 0.0)  # p3 top-right
        lm[3] = _make_landmark(1.0, 0.5)   # p4
        lm[4] = _make_landmark(0.75, 1.0)  # p5 bottom-right
        lm[5] = _make_landmark(0.25, 1.0)  # p6 bottom-left

        ear = eye_aspect_ratio(lm, indices, W, H)
        # For a circular eye: EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
        # ≈ (sqrt((25-25)²+(0-100)²) + sqrt((75-75)²+(0-100)²)) / (2*100)
        # = (100 + 100) / 200 = 1.0
        assert ear > EAR_THRESH, f"Open eye EAR={ear} should be > {EAR_THRESH}"

    def test_closed_eye(self):
        """When vertical distances collapse to 0, EAR ≈ 0."""
        lm = _landmark_list()
        W, H = 100, 100
        indices = [0, 1, 2, 3, 4, 5]
        # Completely flat eye: all y = 0.5
        lm[0] = _make_landmark(0.0, 0.5)
        lm[1] = _make_landmark(0.25, 0.5)
        lm[2] = _make_landmark(0.75, 0.5)
        lm[3] = _make_landmark(1.0, 0.5)
        lm[4] = _make_landmark(0.75, 0.5)
        lm[5] = _make_landmark(0.25, 0.5)

        ear = eye_aspect_ratio(lm, indices, W, H)
        assert ear < EAR_THRESH, f"Closed eye EAR={ear} should be < {EAR_THRESH}"

    def test_ear_returns_float(self):
        lm = _landmark_list()
        ear = eye_aspect_ratio(lm, [0, 1, 2, 3, 4, 5], 640, 480)
        assert isinstance(ear, float)

    def test_ear_non_negative(self):
        lm = _landmark_list()
        ear = eye_aspect_ratio(lm, [0, 1, 2, 3, 4, 5], 640, 480)
        assert ear >= 0.0

    def test_ear_symmetry(self):
        """EAR of a horizontally mirrored eye should be the same."""
        lm1 = _landmark_list()
        lm2 = _landmark_list()
        W, H = 100, 100
        idx = [0, 1, 2, 3, 4, 5]
        coords = [(0.0, 0.5), (0.25, 0.2), (0.75, 0.2),
                   (1.0, 0.5), (0.75, 0.8), (0.25, 0.8)]
        for i, (x, y) in enumerate(coords):
            lm1[i] = _make_landmark(x, y)
            lm2[i] = _make_landmark(1 - x, y)   # horizontal mirror

        ear1 = eye_aspect_ratio(lm1, idx, W, H)
        ear2 = eye_aspect_ratio(lm2, idx, W, H)
        assert abs(ear1 - ear2) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# MAR tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMAR:
    def _make_mouth_landmarks(self, vert_open: float, horiz_width: float):
        """
        Build a minimal landmark list with mouth landmarks set for given
        vertical opening and horizontal width (all in [0,1] scale).
        """
        from src.features import MOUTH_VERT, MOUTH_HORIZ
        lm = _landmark_list()
        cx, cy = 0.5, 0.5
        lm[MOUTH_VERT[0]] = _make_landmark(cx, cy - vert_open / 2)
        lm[MOUTH_VERT[1]] = _make_landmark(cx, cy + vert_open / 2)
        lm[MOUTH_HORIZ[0]] = _make_landmark(cx - horiz_width / 2, cy)
        lm[MOUTH_HORIZ[1]] = _make_landmark(cx + horiz_width / 2, cy)
        return lm

    def test_yawning_mouth(self):
        """Wide-open mouth → MAR > MAR_THRESH."""
        W, H = 100, 100
        lm = self._make_mouth_landmarks(vert_open=0.5, horiz_width=0.3)
        mar = mouth_aspect_ratio(lm, W, H)
        assert mar > MAR_THRESH, f"Yawning MAR={mar} should be > {MAR_THRESH}"

    def test_closed_mouth(self):
        """Closed mouth → MAR < MAR_THRESH."""
        W, H = 100, 100
        lm = self._make_mouth_landmarks(vert_open=0.02, horiz_width=0.3)
        mar = mouth_aspect_ratio(lm, W, H)
        assert mar < MAR_THRESH, f"Closed MAR={mar} should be < {MAR_THRESH}"

    def test_mar_returns_float(self):
        W, H = 640, 480
        lm = self._make_mouth_landmarks(0.1, 0.2)
        mar = mouth_aspect_ratio(lm, W, H)
        assert isinstance(mar, float)

    def test_mar_non_negative(self):
        W, H = 640, 480
        lm = self._make_mouth_landmarks(0.1, 0.2)
        assert mouth_aspect_ratio(lm, W, H) >= 0.0

    def test_mar_increases_with_opening(self):
        """Larger vertical opening → larger MAR."""
        W, H = 100, 100
        lm_small = self._make_mouth_landmarks(vert_open=0.05, horiz_width=0.3)
        lm_large = self._make_mouth_landmarks(vert_open=0.40, horiz_width=0.3)
        assert mouth_aspect_ratio(lm_large, W, H) > mouth_aspect_ratio(lm_small, W, H)


# ──────────────────────────────────────────────────────────────────────────────
# Head pose tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHeadPose:
    def test_returns_three_floats(self):
        lm = _landmark_list()
        from src.features import HEAD_POINTS_IDX, MODEL_3D
        # Spread the head-pose landmarks slightly so solvePnP has something to work with
        for i, idx in enumerate(HEAD_POINTS_IDX):
            lm[idx] = _make_landmark(
                0.3 + (i % 3) * 0.15,
                0.3 + (i // 3) * 0.2,
            )
        result = head_pose_angles(lm, 640, 480)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_no_crash_on_degenerate_landmarks(self):
        """All landmarks at same point → solvePnP may fail → returns (0,0,0)."""
        lm = _landmark_list()
        result = head_pose_angles(lm, 640, 480)
        assert isinstance(result, tuple) and len(result) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Threshold sanity
# ──────────────────────────────────────────────────────────────────────────────

class TestThresholds:
    def test_ear_thresh_range(self):
        assert 0.15 <= EAR_THRESH <= 0.35

    def test_mar_thresh_range(self):
        assert 0.40 <= MAR_THRESH <= 0.80

    def test_head_pitch_thresh_positive(self):
        assert HEAD_PITCH_THRESH > 0
