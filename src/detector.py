"""
Real-time drowsiness detector.

Processes webcam or video file, overlays EAR/MAR/head-pose stats,
and emits an audio beep alert when drowsiness is detected.

Detection logic:
  DROWSY if any of:
    - EAR below threshold for CONSEC_FRAMES_EAR consecutive frames
    - MAR above threshold for CONSEC_FRAMES_MAR consecutive frames
    - Head pitch (nose-down nod) exceeds HEAD_PITCH_THRESH for 10 frames
    - CNN drowsiness model (if available) scores ≥ 0.65 for 5 frames
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.config import (EAR_THRESH, MAR_THRESH, HEAD_PITCH_THRESH,
                         CONSEC_FRAMES_EAR, CONSEC_FRAMES_MAR,
                         IMG_SIZE, MODELS_DIR)
from src.features import (eye_aspect_ratio, mouth_aspect_ratio,
                           head_pose_angles, LEFT_EYE, RIGHT_EYE)
from src.model import DrowsyCNN
from src.dataset import get_transform

FACE_LANDMARKER_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")

# ──────────────────────────────────────────────────────────────────────────────
# Alert (beep)
# ──────────────────────────────────────────────────────────────────────────────

_alert_lock = threading.Lock()
_alert_active = False


def _beep_thread():
    try:
        import sounddevice as sd
        t = np.linspace(0, 0.4, int(44100 * 0.4), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
        sd.play(tone, samplerate=44100)
        sd.wait()
    except Exception:
        # Fallback: terminal bell
        print("\a", end="", flush=True)
    global _alert_active
    with _alert_lock:
        _alert_active = False


def trigger_alert():
    global _alert_active
    with _alert_lock:
        if _alert_active:
            return
        _alert_active = True
    threading.Thread(target=_beep_thread, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_cnn(task: str, device):
    path = os.path.join(MODELS_DIR, f"{task}_best.pt")
    if not os.path.exists(path):
        return None
    model = DrowsyCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Detector class
# ──────────────────────────────────────────────────────────────────────────────

class DrowsinessDetector:
    """
    Wraps MediaPipe face mesh + CNN models for frame-by-frame analysis.
    """

    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.eye_model   = _load_cnn("eye",   self.device)
        self.mouth_model = _load_cnn("mouth",  self.device)
        self.drowsy_model = _load_cnn("drowsy", self.device)
        self.transform = get_transform(augment=False)

        base_options = mp_python.BaseOptions(
            model_asset_path=FACE_LANDMARKER_MODEL
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_mesh = mp_vision.FaceLandmarker.create_from_options(options)

        # Rolling counters
        self.ear_counter  = 0
        self.mar_counter  = 0
        self.pitch_counter = 0
        self.cnn_counter  = 0

        # Last computed values (for display even if no face)
        self.last_ear   = 0.0
        self.last_mar   = 0.0
        self.last_pitch = 0.0
        self.last_cnn_score = 0.0

    def _cnn_predict(self, model, face_roi):
        """Run a loaded CNN on a BGR face crop. Returns prob of class=1."""
        if model is None or face_roi is None:
            return 0.0
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        probs = model.predict_proba(tensor)
        return probs[0, 1].item()

    def process_frame(self, frame):
        """
        Analyse one BGR frame.

        Returns (annotated_frame, is_drowsy).
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_mesh.detect(mp_image)

        is_drowsy = False
        reasons   = []

        if results.face_landmarks:
            lm = results.face_landmarks[0]

            # EAR
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            self.last_ear = ear

            if ear < EAR_THRESH:
                self.ear_counter += 1
                if self.ear_counter >= CONSEC_FRAMES_EAR:
                    is_drowsy = True
                    reasons.append("EYES CLOSED")
            else:
                self.ear_counter = 0

            # MAR
            mar = mouth_aspect_ratio(lm, w, h)
            self.last_mar = mar

            if mar > MAR_THRESH:
                self.mar_counter += 1
                if self.mar_counter >= CONSEC_FRAMES_MAR:
                    is_drowsy = True
                    reasons.append("YAWNING")
            else:
                self.mar_counter = 0

            # Head pose
            pitch, yaw, roll = head_pose_angles(lm, w, h)
            self.last_pitch = pitch

            if pitch > HEAD_PITCH_THRESH:
                self.pitch_counter += 1
                if self.pitch_counter >= 10:
                    is_drowsy = True
                    reasons.append("NODDING")
            else:
                self.pitch_counter = 0

            # CNN drowsiness (whole-face)
            if self.drowsy_model is not None:
                xs = [lm[i].x * w for i in range(468)]
                ys = [lm[i].y * h for i in range(468)]
                x1, y1 = max(0, int(min(xs)) - 10), max(0, int(min(ys)) - 10)
                x2, y2 = min(w, int(max(xs)) + 10), min(h, int(max(ys)) + 10)
                face_roi = frame[y1:y2, x1:x2]
                score = self._cnn_predict(self.drowsy_model, face_roi)
                self.last_cnn_score = score
                if score >= 0.65:
                    self.cnn_counter += 1
                    if self.cnn_counter >= 5:
                        is_drowsy = True
                        reasons.append(f"CNN({score:.2f})")
                else:
                    self.cnn_counter = 0

        # ── Draw overlay ──────────────────────────────────────────
        overlay = frame.copy()
        alpha = 0.0

        if is_drowsy:
            trigger_alert()
            # Red flash
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            alpha = 0.3

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        status_text = "DROWSY!" if is_drowsy else "ALERT"
        cv2.putText(frame, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.putText(frame, f"EAR: {self.last_ear:.3f}  (thr {EAR_THRESH})",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {self.last_mar:.3f}  (thr {MAR_THRESH})",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {self.last_pitch:.1f} deg",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if self.drowsy_model:
            cv2.putText(frame, f"CNN: {self.last_cnn_score:.3f}",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if reasons:
            cv2.putText(frame, " + ".join(reasons), (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, is_drowsy


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(source=0):
    """
    source: 0 for webcam, or a video file path.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}", file=sys.stderr)
        sys.exit(1)

    detector = DrowsinessDetector()
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, is_drowsy = detector.process_frame(frame)
        cv2.imshow("Drowsiness Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="Webcam index (0) or path to video file")
    args = p.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    run(src)
