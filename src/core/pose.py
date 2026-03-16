"""
Advanced Pose Estimator with full-body multi-joint kinematic analysis.

Uses MediaPipe BlazePose to detect 33 body landmarks, then computes
angles for all major joints (elbows, knees, shoulders, hips, neck),
per-joint velocity, body bounding box, and a visibility-weighted
confidence score. Values are passed through EMA filters for stability.

Also supports real-time background segmentation (blur or color replace)
using BlazePose's built-in segmentation mask.
"""

import cv2
import numpy as np
import mediapipe as mp

from src.utils.angles import calculate_angle_2d, euclidean_distance_2d
from src.utils.smoothing import EMAFilterBank


# MediaPipe Pose landmark indices for reference:
#   0  = Nose
#  11  = Left Shoulder    12  = Right Shoulder
#  13  = Left Elbow       14  = Right Elbow
#  15  = Left Wrist       16  = Right Wrist
#  23  = Left Hip         24  = Right Hip
#  25  = Left Knee        26  = Right Knee
#  27  = Left Ankle       28  = Right Ankle

# Joint definitions: (name, point_A, vertex_B, point_C)
JOINT_ANGLES = [
    ('right_elbow',    12, 14, 16),
    ('left_elbow',     11, 13, 15),
    ('right_shoulder', 14, 12, 24),
    ('left_shoulder',  13, 11, 23),
    ('right_knee',     24, 26, 28),
    ('left_knee',      23, 25, 27),
    ('right_hip',      12, 24, 26),
    ('left_hip',       11, 23, 25),
    ('neck',           11, 0,  12),  # approximation: shoulders -> nose
]

# Colors for angle label drawing (BGR)
ANGLE_COLORS = {
    'right_elbow':    (0, 100, 255),
    'left_elbow':     (255, 100, 0),
    'right_shoulder': (0, 200, 200),
    'left_shoulder':  (200, 200, 0),
    'right_knee':     (0, 0, 255),
    'left_knee':      (255, 0, 0),
    'right_hip':      (0, 180, 180),
    'left_hip':       (180, 180, 0),
    'neck':           (200, 200, 200),
}


class PoseTracker:
    """
    Full-body pose estimator with multi-joint kinematics.

    Provides:
    - 33-landmark detection with MediaPipe BlazePose
    - Multi-joint angle computation (9 joints)
    - Per-joint velocity tracking (pixels/frame displacement)
    - Body bounding box
    - Visibility-weighted confidence score
    - EMA-smoothed angle output
    - Real-time background segmentation (blur/color)
    """

    def __init__(self, mode=False, model_complex=1, smooth_landmarks=True,
                 enable_seg=True, smooth_seg=True,
                 detection_con=0.5, track_con=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self._seg_enabled = enable_seg
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=model_complex,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_seg,
            smooth_segmentation=smooth_seg,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )

        self.results = None
        self._prev_landmarks = {}
        self._angle_filters = EMAFilterBank(alpha=0.35)
        self._seg_mode = 0  # 0=off, 1=blur, 2=color replace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, img, draw=True):
        """Run pose detection on a BGR frame. Optionally draws skeleton."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        return img

    def get_landmarks(self, img):
        """
        Extract landmarks into a dict {id: (cx, cy, cz, visibility)}.
        Coordinates are in pixel space for x,y and normalized z.
        """
        landmarks = {}
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[idx] = (cx, cy, lm.z, lm.visibility)
        return landmarks

    def compute_all_angles(self, landmarks, img=None, draw=False):
        """
        Compute smoothed angles for all defined joints.

        Returns dict {joint_name: smoothed_angle_degrees}.
        If draw=True and img is provided, annotates the angle on the frame.
        """
        angles = {}
        for name, p1, p2, p3 in JOINT_ANGLES:
            if p1 in landmarks and p2 in landmarks and p3 in landmarks:
                raw = calculate_angle_2d(
                    landmarks[p1][:2], landmarks[p2][:2], landmarks[p3][:2]
                )
                smoothed = self._angle_filters.update(name, raw)
                angles[name] = round(smoothed, 1)

                if draw and img is not None:
                    self._draw_angle_label(img, name, smoothed, landmarks[p2][:2])
            else:
                angles[name] = None
        return angles

    def compute_velocities(self, landmarks):
        """
        Compute per-joint velocity as pixel displacement since last frame.

        Returns dict {landmark_id: velocity_px}.
        """
        velocities = {}
        for idx, (cx, cy, _, _) in landmarks.items():
            if idx in self._prev_landmarks:
                px, py = self._prev_landmarks[idx][:2]
                velocities[idx] = round(euclidean_distance_2d((cx, cy), (px, py)), 1)
            else:
                velocities[idx] = 0.0

        # Store current as previous for next frame
        self._prev_landmarks = {idx: vals[:2] for idx, vals in landmarks.items()}
        return velocities

    def compute_body_bbox(self, landmarks):
        """
        Return the axis-aligned bounding box of all visible landmarks.

        Returns (x_min, y_min, x_max, y_max) or None.
        """
        if not landmarks:
            return None
        xs = [v[0] for v in landmarks.values()]
        ys = [v[1] for v in landmarks.values()]
        return (min(xs), min(ys), max(xs), max(ys))

    def compute_confidence(self, landmarks):
        """
        Compute a visibility-weighted confidence score [0, 100].

        Averages the visibility of all detected landmarks and scales to
        a percentage.
        """
        if not landmarks:
            return 0.0
        visibilities = [v[3] for v in landmarks.values()]
        return round(sum(visibilities) / len(visibilities) * 100, 1)

    def get_full_pose_data(self, img, draw=False):
        """
        Convenience method: runs all pose analytics in one call.

        Returns dict with keys:
        - landmarks, angles, velocities, bbox, confidence
        """
        landmarks = self.get_landmarks(img)
        if not landmarks:
            return {
                'landmarks': {},
                'angles': {},
                'velocities': {},
                'bbox': None,
                'confidence': 0.0,
            }

        angles = self.compute_all_angles(landmarks, img=img, draw=draw)
        velocities = self.compute_velocities(landmarks)
        bbox = self.compute_body_bbox(landmarks)
        confidence = self.compute_confidence(landmarks)
        posture = self._check_spine_alignment(landmarks)

        return {
            'landmarks': landmarks,
            'angles': angles,
            'velocities': velocities,
            'bbox': bbox,
            'confidence': confidence,
            'posture': posture,
        }

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_angle_label(self, img, name, angle, vertex_xy):
        """Draw a small color-coded angle label near the joint vertex."""
        color = ANGLE_COLORS.get(name, (255, 255, 255))
        label = f"{int(angle)}"
        x, y = vertex_xy

        # Offset so labels don't overlap the joint dot
        offset_x, offset_y = 15, -10
        if 'left' in name:
            offset_x = -60

        cv2.putText(
            img, label, (x + offset_x, y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    def _check_spine_alignment(self, landmarks):
        """
        Check for spine straightness using shoulder-hip alignment.
        Useful for planks, squats, and sitting posture.
        """
        if not (11 in landmarks and 12 in landmarks and 23 in landmarks and 24 in landmarks):
            return "Incomplete Data"

        # Center points
        shoulder_mid = ((landmarks[11][0] + landmarks[12][0]) // 2, 
                        (landmarks[11][1] + landmarks[12][1]) // 2)
        hip_mid = ((landmarks[23][0] + landmarks[24][0]) // 2, 
                   (landmarks[23][1] + landmarks[24][1]) // 2)

        # Vector from hip to shoulder
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        
        # In sitting/standing, dy is usually much larger than dx (spine is vertical)
        # In plank, dx is larger (spine is horizontal)
        
        # Calculate angle from horizontal
        spine_angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        
        # Simple heuristic: for standing/sitting, spine should be near 90deg (vertical)
        # If it's less than 65deg, user might be slouching significantly forward.
        if spine_angle < 65:
            return "Slouching"
        return "Straight"

    # ------------------------------------------------------------------
    # Background Segmentation
    # ------------------------------------------------------------------

    @property
    def seg_mode(self):
        """Current segmentation mode: 0=off, 1=blur, 2=color."""
        return self._seg_mode

    @property
    def seg_mode_label(self):
        """Human-readable segmentation mode label."""
        return ['Off', 'Blur', 'Color'][self._seg_mode]

    def cycle_seg_mode(self):
        """Cycle through segmentation modes: Off -> Blur -> Color -> Off."""
        self._seg_mode = (self._seg_mode + 1) % 3

    def apply_segmentation(self, img):
        """
        Apply background segmentation to the frame based on current mode.

        Mode 0: No effect (passthrough)
        Mode 1: Gaussian blur on background
        Mode 2: Replace background with a dark gradient

        Returns the processed image.
        """
        if self._seg_mode == 0 or not self._seg_enabled:
            return img

        if not (self.results and self.results.segmentation_mask is not None):
            return img

        mask = self.results.segmentation_mask
        # Threshold the mask for a clean binary segmentation
        condition = mask > 0.5

        if self._seg_mode == 1:
            # Blur background
            blurred = cv2.GaussianBlur(img, (55, 55), 0)
            output = np.where(condition[:, :, np.newaxis], img, blurred)

        elif self._seg_mode == 2:
            # Color gradient background
            h, w, _ = img.shape
            bg = np.zeros_like(img)
            # Create a dark blue-purple gradient
            for row in range(h):
                ratio = row / h
                bg[row, :] = (
                    int(40 + 20 * ratio),   # B
                    int(20 + 10 * ratio),   # G
                    int(30 + 40 * ratio),   # R
                )
            output = np.where(condition[:, :, np.newaxis], img, bg)
        else:
            return img

        return output.astype(np.uint8)
