"""
Premium multi-panel HUD dashboard for real-time analytics visualization.

Draws a semi-transparent overlay with color-coded panels for pose analytics,
hand analytics, face analytics, exercise reps, symmetry scoring, activity
status, and system info. Designed for 1280x720 resolution but adapts to
any frame size.
"""

import cv2
import time


# Panel color scheme (BGR)
COLOR_BG = (20, 20, 20)
COLOR_HEADER = (220, 220, 220)
COLOR_LABEL = (180, 180, 180)
COLOR_POSE = (100, 200, 255)     # orange-ish
COLOR_HAND = (255, 255, 100)     # cyan-ish
COLOR_FACE = (150, 255, 150)     # green
COLOR_ACTIVITY = (180, 150, 255) # pink/magenta
COLOR_EXERCISE = (100, 255, 255) # yellow
COLOR_SYMMETRY = (255, 200, 180) # light blue
COLOR_RECOLOR_BAD = (0, 0, 0) # black

# Pose connections for neon drawing
POSE_CONNECTIONS = [
    (11, 12), # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # torso
    (23, 24), # hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28), # right leg
]

# Joint point indices for neon highlights
POSE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COLOR_RECORDING = (0, 0, 255)    # red
COLOR_FPS = (0, 255, 0)          # green
COLOR_DIM = (120, 120, 120)
COLOR_GOOD = (0, 255, 0)
COLOR_WARN = (0, 180, 255)
COLOR_BAD = (0, 0, 255)

# Layout constants
PANEL_WIDTH = 310
PANEL_MARGIN = 10
PANEL_ALPHA = 0.55
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    """
    Premium multi-panel HUD overlay with 7+ analytics panels.

    Panels (right side, top to bottom):
    1. Pose Analytics   (joint angles, confidence)
    2. Hand Analytics   (gestures, openness, pinch)
    3. Face Analytics   (EAR, MAR, expression, tilt)
    4. Exercise Reps    (per-exercise rep count and phase)
    5. Symmetry Score   (bilateral comparison)
    6. Activity Status  (current posture)

    System bar (top-left):
    - FPS, recording indicators, module toggles, control hints
    """

    def __init__(self):
        self._p_time = 0
        self._fps = 0
        self._fps_smooth = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def draw_hud(self, img, pose_data, hand_data, face_data,
                 activity_label, exercise_data, symmetry_data,
                 modules_active, is_recording, is_video_recording,
                 seg_mode_label, trails_on, graphs_on):
        """
        Draw the full HUD overlay on the frame.

        Parameters
        ----------
        img : BGR frame
        pose_data : dict from PoseTracker.get_full_pose_data()
        hand_data : list from GestureTracker.get_full_hand_data()
        face_data : dict from FaceTracker.get_face_data()
        activity_label : str from ActivityClassifier
        exercise_data : list from ExerciseCounter.get_exercise_data()
        symmetry_data : dict from SymmetryAnalyzer.analyze()
        modules_active : dict {'pose','hands','face': bool}
        is_recording : bool (CSV)
        is_video_recording : bool (MP4)
        seg_mode_label : str
        trails_on : bool
        graphs_on : bool
        """
        h, w, _ = img.shape
        self._update_fps()

        # Right-side panels
        x_start = w - PANEL_WIDTH - PANEL_MARGIN
        y_cursor = PANEL_MARGIN

        # Panel 1: Pose
        if modules_active.get('pose', True):
            y_cursor = self._draw_pose_panel(img, x_start, y_cursor, pose_data)
            y_cursor += 4

        # Panel 2: Hands
        if modules_active.get('hands', True):
            y_cursor = self._draw_hand_panel(img, x_start, y_cursor, hand_data)
            y_cursor += 4

        # Panel 3: Face
        if modules_active.get('face', True):
            y_cursor = self._draw_face_panel(img, x_start, y_cursor, face_data)
            y_cursor += 4

        # Panel 4: Exercise Reps
        if exercise_data:
            y_cursor = self._draw_exercise_panel(img, x_start, y_cursor, exercise_data)
            y_cursor += 4

        # Panel 5: Symmetry
        if symmetry_data and symmetry_data.get('pairs'):
            y_cursor = self._draw_symmetry_panel(img, x_start, y_cursor, symmetry_data)
            y_cursor += 4

        # Panel 6: Activity
        y_cursor = self._draw_activity_panel(img, x_start, y_cursor, activity_label)

        # Top-left system bar
        self._draw_system_bar(
            img, modules_active, is_recording, is_video_recording,
            seg_mode_label, trails_on, graphs_on,
        )

        return img

    def draw_neon_skeleton(self, img, landmarks, posture="Straight"):
        """
        Draw a custom neon-styled skeleton with glow effects.
        
        Parameters
        ----------
        img : BGR frame
        landmarks : dict {id: (x, y, z, vis)}
        posture : str ('Straight'/'Slouching')
        """
        if not landmarks:
            return img

        # Draw connections
        bone_color = (255, 255, 255) # white core
        glow_color = (255, 100, 0)   # cyan glow
        
        if posture == "Slouching":
            glow_color = (0, 100, 255) # orange/red alert
        
        # 1. Draw outer glow (thick, transparent)
        glow_img = img.copy()
        for p1_id, p2_id in POSE_CONNECTIONS:
            if p1_id in landmarks and p2_id in landmarks:
                p1 = landmarks[p1_id][:2]
                p2 = landmarks[p2_id][:2]
                if landmarks[p1_id][3] > 0.5 and landmarks[p2_id][3] > 0.5:
                    cv2.line(glow_img, p1, p2, glow_color, 8, cv2.LINE_AA)
        
        # 2. Add weighted glow to main image
        cv2.addWeighted(glow_img, 0.4, img, 0.6, 0, img)
        
        # 3. Draw inner bones (thin, bright)
        for p1_id, p2_id in POSE_CONNECTIONS:
            if p1_id in landmarks and p2_id in landmarks:
                p1 = landmarks[p1_id][:2]
                p2 = landmarks[p2_id][:2]
                if landmarks[p1_id][3] > 0.5 and landmarks[p2_id][3] > 0.5:
                    cv2.line(img, p1, p2, bone_color, 2, cv2.LINE_AA)
                    
        # 4. Draw joints
        for j_id in POSE_JOINTS:
            if j_id in landmarks and landmarks[j_id][3] > 0.5:
                cv2.circle(img, landmarks[j_id][:2], 4, bone_color, -1, cv2.LINE_AA)
                cv2.circle(img, landmarks[j_id][:2], 6, glow_color, 1, cv2.LINE_AA)

        return img

    # ------------------------------------------------------------------
    # Individual panels
    # ------------------------------------------------------------------

    def _draw_pose_panel(self, img, x, y, pose_data):
        """Draw Pose Analytics panel."""
        angles = pose_data.get('angles', {})
        confidence = pose_data.get('confidence', 0)

        angle_lines = [(k, v) for k, v in angles.items() if v is not None]
        num_lines = 2 + len(angle_lines)
        panel_h = 24 + num_lines * 18

        self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
        self._put_text(img, "POSE ANALYTICS", x + 10, y + 18, 0.45, COLOR_HEADER, 1)

        cy = y + 38
        conf_color = COLOR_POSE if confidence > 50 else COLOR_DIM
        self._put_text(img, f"Confidence: {confidence:.0f}%", x + 10, cy, 0.38, conf_color)
        cy += 18

        for name, val in angle_lines:
            label = name.replace('_', ' ').title()
            self._put_text(img, f"{label}: {val:.0f}", x + 10, cy, 0.35, COLOR_POSE)
            cy += 18

        return y + panel_h

    def _draw_hand_panel(self, img, x, y, hand_data):
        """Draw Hand Analytics panel."""
        if not hand_data:
            panel_h = 44
            self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
            self._put_text(img, "HAND ANALYTICS", x + 10, y + 18, 0.45, COLOR_HEADER, 1)
            self._put_text(img, "No hands detected", x + 10, y + 36, 0.35, COLOR_DIM)
            return y + panel_h

        num_lines = 1
        for _ in hand_data:
            num_lines += 3
        panel_h = 24 + num_lines * 18

        self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
        self._put_text(img, "HAND ANALYTICS", x + 10, y + 18, 0.45, COLOR_HEADER, 1)

        cy = y + 38
        for hd in hand_data:
            self._put_text(img, f"{hd['type']}: {hd['gesture']}", x + 10, cy, 0.38, COLOR_HAND)
            cy += 18
            self._put_text(img, f"  Open: {hd.get('openness',0):.0f}%  Pinch: {hd.get('pinch_distance',0):.0f}px", x + 10, cy, 0.32, COLOR_LABEL)
            cy += 18

        return y + panel_h

    def _draw_face_panel(self, img, x, y, face_data):
        """Draw Face Analytics panel."""
        if not face_data:
            panel_h = 44
            self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
            self._put_text(img, "FACE ANALYTICS", x + 10, y + 18, 0.45, COLOR_HEADER, 1)
            self._put_text(img, "No face detected", x + 10, y + 36, 0.35, COLOR_DIM)
            return y + panel_h

        panel_h = 24 + 5 * 18
        self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
        self._put_text(img, "FACE ANALYTICS", x + 10, y + 18, 0.45, COLOR_HEADER, 1)

        cy = y + 38
        self._put_text(img, f"Expression: {face_data.get('expression','N/A')}", x + 10, cy, 0.38, COLOR_FACE)
        cy += 18
        self._put_text(img, f"EAR: {face_data.get('ear_avg',0):.3f}  MAR: {face_data.get('mar',0):.3f}", x + 10, cy, 0.35, COLOR_LABEL)
        cy += 18
        self._put_text(img, f"Head Tilt: {face_data.get('head_tilt',0):.1f} deg", x + 10, cy, 0.35, COLOR_LABEL)
        cy += 18
        self._put_text(img, f"EAR L:{face_data.get('ear_left',0):.3f} R:{face_data.get('ear_right',0):.3f}", x + 10, cy, 0.32, COLOR_DIM)

        return y + panel_h

    def _draw_exercise_panel(self, img, x, y, exercise_data):
        """Draw Exercise Rep Counter panel."""
        active_exercises = [e for e in exercise_data if e.get('active')]
        if not active_exercises:
            # Show compact panel
            panel_h = 44
            self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
            self._put_text(img, "EXERCISE TRACKER", x + 10, y + 18, 0.45, COLOR_HEADER, 1)
            total = sum(e.get('reps', 0) for e in exercise_data)
            self._put_text(img, f"Total Reps: {total}  (waiting...)", x + 10, y + 36, 0.35, COLOR_DIM)
            return y + panel_h

        num_lines = 1 + len(active_exercises)
        panel_h = 24 + num_lines * 20

        self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
        self._put_text(img, "EXERCISE TRACKER", x + 10, y + 18, 0.45, COLOR_HEADER, 1)

        cy = y + 40
        for ex in active_exercises:
            phase_color = COLOR_WARN if ex['phase'] == 'Down' else COLOR_EXERCISE
            self._put_text(
                img,
                f"{ex['label']}: {ex['reps']} reps [{ex['phase']}]",
                x + 10, cy, 0.38, phase_color,
            )
            cy += 20

        return y + panel_h

    def _draw_symmetry_panel(self, img, x, y, symmetry_data):
        """Draw Symmetry Score panel."""
        pairs = symmetry_data.get('pairs', [])
        overall = symmetry_data.get('overall_score', 0)

        valid_pairs = [p for p in pairs if p.get('symmetry_pct') is not None]
        panel_h = 24 + (2 + len(valid_pairs)) * 18

        self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
        self._put_text(img, "SYMMETRY ANALYSIS", x + 10, y + 18, 0.45, COLOR_HEADER, 1)

        cy = y + 38
        # Overall score with color coding
        if overall >= 85:
            score_color = COLOR_GOOD
        elif overall >= 60:
            score_color = COLOR_WARN
        else:
            score_color = COLOR_BAD
        self._put_text(img, f"Overall: {overall:.0f}%", x + 10, cy, 0.42, score_color, 1)
        cy += 20

        for p in valid_pairs:
            self._put_text(
                img,
                f"{p['name']}: L={p['left_angle']:.0f} R={p['right_angle']:.0f} ({p['symmetry_pct']:.0f}%)",
                x + 10, cy, 0.32, COLOR_SYMMETRY,
            )
            cy += 18

        return y + panel_h

    def _draw_activity_panel(self, img, x, y, activity_label):
        """Draw Activity Status panel."""
        panel_h = 44
        self._draw_panel_bg(img, x, y, PANEL_WIDTH, panel_h)
        self._put_text(img, "ACTIVITY", x + 10, y + 18, 0.45, COLOR_HEADER, 1)
        self._put_text(img, activity_label, x + 10, y + 38, 0.5, COLOR_ACTIVITY, 1)
        return y + panel_h

    def _draw_system_bar(self, img, modules_active, is_recording,
                         is_video_recording, seg_mode_label, trails_on, graphs_on):
        """Draw top-left system info bar."""
        bar_h = 82
        bar_w = 420

        self._draw_panel_bg(img, PANEL_MARGIN, PANEL_MARGIN, bar_w, bar_h)

        # Row 1: FPS + Recording
        self._put_text(
            img, f"FPS: {int(self._fps_smooth)}",
            PANEL_MARGIN + 10, PANEL_MARGIN + 20, 0.5, COLOR_FPS, 1,
        )

        rx = PANEL_MARGIN + 100
        if is_recording:
            cv2.circle(img, (rx, PANEL_MARGIN + 14), 5, COLOR_RECORDING, cv2.FILLED)
            self._put_text(img, "CSV", rx + 10, PANEL_MARGIN + 20, 0.4, COLOR_RECORDING, 1)
            rx += 55
        if is_video_recording:
            cv2.circle(img, (rx, PANEL_MARGIN + 14), 5, (0, 100, 255), cv2.FILLED)
            self._put_text(img, "VID", rx + 10, PANEL_MARGIN + 20, 0.4, (0, 100, 255), 1)

        # Row 2: Module toggles
        ty = PANEL_MARGIN + 42
        tx = PANEL_MARGIN + 10
        for key, label in [('pose', 'P:Pose'), ('hands', 'H:Hand'), ('face', 'F:Face')]:
            color = COLOR_FPS if modules_active.get(key, True) else COLOR_DIM
            self._put_text(img, label, tx, ty, 0.34, color)
            tx += 72

        # Trails, Graphs, Seg
        trail_c = COLOR_FPS if trails_on else COLOR_DIM
        self._put_text(img, f"T:Trail", tx, ty, 0.34, trail_c)
        tx += 68
        graph_c = COLOR_FPS if graphs_on else COLOR_DIM
        self._put_text(img, f"G:Graph", tx, ty, 0.34, graph_c)

        # Row 3: More controls
        ty2 = PANEL_MARGIN + 62
        tx2 = PANEL_MARGIN + 10
        self._put_text(img, f"B:Seg[{seg_mode_label}]", tx2, ty2, 0.34, COLOR_LABEL)
        tx2 += 110
        self._put_text(img, "R:Rec  V:Vid  S:Snap  X:Reset  Q:Quit", tx2, ty2, 0.3, COLOR_DIM)

        # Row 4 is intentionally absent to keep bar compact

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_panel_bg(self, img, x, y, w, h):
        """Draw a semi-transparent dark rectangle."""
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BG, cv2.FILLED)
        cv2.addWeighted(overlay, PANEL_ALPHA, img, 1 - PANEL_ALPHA, 0, img)

    def _put_text(self, img, text, x, y, scale, color, thickness=1):
        """Draw anti-aliased text."""
        cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

    def _update_fps(self):
        """Compute FPS with EMA smoothing."""
        c_time = time.time()
        dt = c_time - self._p_time
        if dt > 0:
            self._fps = 1.0 / dt
        self._p_time = c_time
        self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * self._fps

    # Legacy compat
    def update_fps(self, img):
        self._update_fps()
        cv2.putText(img, f"FPS: {int(self._fps_smooth)}", (20, 50),
                    FONT, 1, COLOR_FPS, 2)
        return img

    def draw_dashboard(self, img, pose_info, hand_info):
        return img
