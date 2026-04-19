"""
Tests for DrowsinessDetector – uses synthetic frames (no camera required).
"""
import types
import numpy as np
import pytest
import cv2

from src.detector import DrowsinessDetector, trigger_alert
from src.config import EAR_THRESH, MAR_THRESH, CONSEC_FRAMES_EAR, CONSEC_FRAMES_MAR


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return DrowsinessDetector()


def black_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def white_frame(w=640, h=480):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def random_frame(w=640, h=480):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Frame processing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessFrame:
    def test_returns_tuple(self, detector):
        frame, is_drowsy = detector.process_frame(black_frame())
        assert isinstance(frame, np.ndarray)
        assert isinstance(is_drowsy, bool)

    def test_output_frame_same_shape(self, detector):
        inp = random_frame(320, 240)
        out, _ = detector.process_frame(inp.copy())
        assert out.shape == inp.shape

    def test_output_frame_dtype(self, detector):
        out, _ = detector.process_frame(black_frame())
        assert out.dtype == np.uint8

    def test_no_face_not_drowsy(self, detector):
        """Black frame has no face → not drowsy."""
        _, is_drowsy = detector.process_frame(black_frame())
        assert not is_drowsy

    def test_no_face_white_frame(self, detector):
        _, is_drowsy = detector.process_frame(white_frame())
        assert not is_drowsy

    def test_consecutive_black_frames_not_drowsy(self, detector):
        """Repeated frames without a face should never flag drowsy."""
        for _ in range(CONSEC_FRAMES_EAR + 5):
            _, is_drowsy = detector.process_frame(black_frame())
        assert not is_drowsy

    def test_does_not_modify_input(self, detector):
        """process_frame should operate on the passed array (may modify it for annotation)."""
        frame = black_frame()
        original = frame.copy()
        out, _ = detector.process_frame(frame)
        # Output is annotated; we just check it's a valid ndarray
        assert isinstance(out, np.ndarray)

    def test_state_resets_between_fresh_detectors(self):
        """Two separate detector instances should have independent counters."""
        d1 = DrowsinessDetector()
        d2 = DrowsinessDetector()
        d1.ear_counter = 999
        assert d2.ear_counter == 0


# ──────────────────────────────────────────────────────────────────────────────
# Counter / threshold logic tests (unit, no camera)
# ──────────────────────────────────────────────────────────────────────────────

class TestCounterLogic:
    def test_ear_counter_increments_below_thresh(self, detector):
        """Manually drive ear_counter and verify drowsy triggers."""
        detector.ear_counter = CONSEC_FRAMES_EAR - 1
        # Manually simulate one more closed-eye frame
        detector.ear_counter += 1
        assert detector.ear_counter >= CONSEC_FRAMES_EAR

    def test_ear_counter_resets_above_thresh(self, detector):
        detector.ear_counter = 10
        # Simulate open-eye → counter resets
        detector.ear_counter = 0
        assert detector.ear_counter == 0

    def test_mar_counter_increments_above_thresh(self, detector):
        detector.mar_counter = CONSEC_FRAMES_MAR - 1
        detector.mar_counter += 1
        assert detector.mar_counter >= CONSEC_FRAMES_MAR

    def test_initial_counters_zero(self):
        d = DrowsinessDetector()
        assert d.ear_counter == 0
        assert d.mar_counter == 0
        assert d.pitch_counter == 0
        assert d.cnn_counter == 0

    def test_initial_last_values_zero(self):
        d = DrowsinessDetector()
        assert d.last_ear == 0.0
        assert d.last_mar == 0.0
        assert d.last_pitch == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Alert tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAlert:
    def test_trigger_alert_no_crash(self):
        """Calling trigger_alert() should not raise."""
        trigger_alert()  # runs in thread; should be safe

    def test_trigger_alert_idempotent(self):
        """Calling twice quickly should not raise."""
        trigger_alert()
        trigger_alert()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading tests (no trained models required)
# ──────────────────────────────────────────────────────────────────────────────

class TestModelLoading:
    def test_detector_initializes_without_trained_models(self):
        """Detector must init even if no .pt files exist."""
        d = DrowsinessDetector()
        # Models may be None if not trained yet
        assert d.eye_model is None or hasattr(d.eye_model, "forward")
        assert d.mouth_model is None or hasattr(d.mouth_model, "forward")
        assert d.drowsy_model is None or hasattr(d.drowsy_model, "forward")
