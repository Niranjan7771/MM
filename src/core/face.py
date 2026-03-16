"""
Face Mesh analysis with expression estimation.

Uses MediaPipe FaceMesh (468 landmarks) to compute:
- Eye Aspect Ratio (EAR) for blink detection
- Mouth Aspect Ratio (MAR) for yawn/talking detection
- Head tilt (roll angle) from eye positions
- Expression classification: Neutral, Blinking, Yawning, Talking, Head Tilt

All ratio outputs are EMA-smoothed for stable display.
"""

import cv2
import math
import mediapipe as mp

from src.utils.angles import euclidean_distance_2d, midpoint
from src.utils.smoothing import EMAFilterBank


# FaceMesh landmark indices for eye and mouth contours
# Left eye (from subject's perspective -- appears on RIGHT side of image)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye (from subject's perspective -- appears on LEFT side of image)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Outer lip contour (vertical pairs for MAR)
UPPER_LIP = [13]
LOWER_LIP = [14]
LEFT_MOUTH = [78]
RIGHT_MOUTH = [308]
# Additional vertical lip points for more accurate MAR
UPPER_LIP_INNER = [82, 312]
LOWER_LIP_INNER = [87, 317]

# Thresholds
EAR_BLINK_THRESHOLD = 0.21
MAR_YAWN_THRESHOLD = 0.7
MAR_TALKING_THRESHOLD = 0.35
HEAD_TILT_THRESHOLD = 12.0  # degrees


class FaceTracker:
    """
    Face mesh processor and expression classifier.

    Provides:
    - 468-landmark face mesh detection
    - Eye Aspect Ratio (EAR) per eye
    - Mouth Aspect Ratio (MAR)
    - Head tilt angle (roll)
    - Expression string classification
    """

    def __init__(self, max_faces=1, refine=True,
                 detection_con=0.5, track_con=0.5):
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )

        self.results = None
        self._filters = EMAFilterBank(alpha=0.4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, img, draw=True):
        """Run face mesh detection. Optionally draws the tesselation."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)

        if self.results.multi_face_landmarks and draw:
            for face_lm in self.results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    img, face_lm,
                    self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_styles
                        .get_default_face_mesh_tesselation_style(),
                )
                # Draw eye and lip contours more prominently
                self.mp_draw.draw_landmarks(
                    img, face_lm,
                    self.mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_styles
                        .get_default_face_mesh_contours_style(),
                )
        return img

    def get_face_data(self, img):
        """
        Compute all face analytics for the first detected face.

        Returns dict:
        {
            'ear_left': float,
            'ear_right': float,
            'ear_avg': float,
            'mar': float,
            'head_tilt': float,  # degrees, positive = tilt right
            'expression': str,
        }
        or empty dict if no face detected.
        """
        if not (self.results and self.results.multi_face_landmarks):
            return {}

        face_lm = self.results.multi_face_landmarks[0]
        h, w, _ = img.shape

        # Extract pixel-space coordinates
        pts = {}
        for idx, lm in enumerate(face_lm.landmark):
            pts[idx] = (int(lm.x * w), int(lm.y * h))

        # Compute raw metrics
        ear_l = self._compute_ear(pts, LEFT_EYE)
        ear_r = self._compute_ear(pts, RIGHT_EYE)
        mar = self._compute_mar(pts)
        tilt = self._compute_head_tilt(pts)

        # Smooth
        ear_l = self._filters.update('ear_l', ear_l)
        ear_r = self._filters.update('ear_r', ear_r)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = self._filters.update('mar', mar)
        tilt = self._filters.update('tilt', tilt)

        expression = self._classify_expression(ear_avg, mar, tilt)

        return {
            'ear_left': round(ear_l, 3),
            'ear_right': round(ear_r, 3),
            'ear_avg': round(ear_avg, 3),
            'mar': round(mar, 3),
            'head_tilt': round(tilt, 1),
            'expression': expression,
        }

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _compute_ear(self, pts, eye_indices):
        """
        Eye Aspect Ratio using the 6-point formula:

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

        Where p1..p6 are the 6 eye contour landmarks ordered:
        p1=lateral, p2=upper-lateral, p3=upper-medial,
        p4=medial, p5=lower-medial, p6=lower-lateral
        """
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in eye_indices]

        vertical_1 = euclidean_distance_2d(p2, p6)
        vertical_2 = euclidean_distance_2d(p3, p5)
        horizontal = euclidean_distance_2d(p1, p4)

        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _compute_mar(self, pts):
        """
        Mouth Aspect Ratio:

        MAR = (sum of vertical lip distances) / (horizontal mouth width)
        """
        # Vertical distances (multiple pairs for accuracy)
        vert_center = euclidean_distance_2d(pts[UPPER_LIP[0]], pts[LOWER_LIP[0]])
        vert_left = euclidean_distance_2d(pts[UPPER_LIP_INNER[0]], pts[LOWER_LIP_INNER[0]])
        vert_right = euclidean_distance_2d(pts[UPPER_LIP_INNER[1]], pts[LOWER_LIP_INNER[1]])

        # Horizontal distance
        horiz = euclidean_distance_2d(pts[LEFT_MOUTH[0]], pts[RIGHT_MOUTH[0]])

        if horiz == 0:
            return 0.0
        return (vert_center + vert_left + vert_right) / (2.0 * horiz)

    def _compute_head_tilt(self, pts):
        """
        Approximate head roll angle from eye center positions.

        Positive angle = tilting head to the right.
        """
        left_eye_center = midpoint(pts[LEFT_EYE[0]], pts[LEFT_EYE[3]])
        right_eye_center = midpoint(pts[RIGHT_EYE[0]], pts[RIGHT_EYE[3]])

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]

        return math.degrees(math.atan2(dy, dx))

    # ------------------------------------------------------------------
    # Expression classification
    # ------------------------------------------------------------------

    def _classify_expression(self, ear_avg, mar, tilt):
        """
        Rule-based expression classification from computed metrics.

        Priority: Blinking > Yawning > Talking > Head Tilt > Neutral
        """
        if ear_avg < EAR_BLINK_THRESHOLD:
            return "Blinking"

        if mar > MAR_YAWN_THRESHOLD:
            return "Yawning"

        if mar > MAR_TALKING_THRESHOLD:
            return "Talking"

        if abs(tilt) > HEAD_TILT_THRESHOLD:
            direction = "Right" if tilt > 0 else "Left"
            return f"Head Tilt {direction}"

        return "Neutral"
