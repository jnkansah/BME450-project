"""
Compute EAR, MAR, and head-pose from MediaPipe face-mesh landmarks.

All functions accept a single-face landmark list (normalized [0,1] coords)
as returned by mediapipe.solutions.face_mesh.
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe face-mesh landmark indices
# ---------------------------------------------------------------------------
# Eye landmarks (6 points per eye, standard MediaPipe indices)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks (outer lip contour subset)
MOUTH = [61, 291, 39, 269, 0, 17, 405, 181]
MOUTH_VERT = [13, 14]    # upper / lower inner lip
MOUTH_HORIZ = [78, 308]  # corners

# Head-pose reference points (for solvePnP)
# Nose tip, chin, left/right eye corners, left/right mouth corners
HEAD_POINTS_IDX = [1, 152, 226, 446, 57, 287]

# Canonical 3-D face model (mm) – standard MediaPipe reference
MODEL_3D = np.array([
    [0.0,    0.0,    0.0],   # nose tip
    [0.0,  -330.0, -65.0],   # chin
    [-225.0, 170.0, -135.0], # left eye corner
    [225.0,  170.0, -135.0], # right eye corner
    [-150.0, -150.0, -125.0],# left mouth corner
    [150.0,  -150.0, -125.0] # right mouth corner
], dtype=np.float64)


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Returns float in [0, ~0.5]; < EAR_THRESH → eye closed.
    """
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    ear = (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)
    return ear


def mouth_aspect_ratio(landmarks, img_w, img_h):
    """
    MAR = vertical mouth opening / horizontal mouth width.
    Returns float; > MAR_THRESH → yawning.
    """
    top = (landmarks[MOUTH_VERT[0]].x * img_w, landmarks[MOUTH_VERT[0]].y * img_h)
    bot = (landmarks[MOUTH_VERT[1]].x * img_w, landmarks[MOUTH_VERT[1]].y * img_h)
    left = (landmarks[MOUTH_HORIZ[0]].x * img_w, landmarks[MOUTH_HORIZ[0]].y * img_h)
    right = (landmarks[MOUTH_HORIZ[1]].x * img_w, landmarks[MOUTH_HORIZ[1]].y * img_h)
    mar = _dist(top, bot) / (_dist(left, right) + 1e-6)
    return mar


def head_pose_angles(landmarks, img_w, img_h):
    """
    Estimate pitch / yaw / roll (degrees) via solvePnP.
    Returns (pitch, yaw, roll); pitch > HEAD_PITCH_THRESH → nodding.
    """
    import cv2
    image_points = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in HEAD_POINTS_IDX
    ], dtype=np.float64)

    focal_length = img_w
    camera_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    try:
        ok, rvec, _ = cv2.solvePnP(
            MODEL_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return 0.0, 0.0, 0.0
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix to Euler angles
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        roll  = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw   = 0.0
        roll  = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))

    return pitch, yaw, roll
