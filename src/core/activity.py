"""
Activity and posture classifier using a temporal smoothing state machine.

Analyzes pose landmark positions to classify full-body posture into states
like Standing, Sitting, Arms Raised, T-Pose, Leaning, etc. Uses a debounce
mechanism (N consecutive frames of same classification) to prevent flickery
label switching. Tracks an activity timeline for session analytics.
"""

import time


# Minimum consecutive frames before the state machine transitions
DEBOUNCE_FRAMES = 8


class ActivityClassifier:
    """
    Posture state machine with temporal debouncing.

    Feed it pose landmark data every frame via `update()` and it returns
    the current stable activity label. Also maintains an activity timeline
    for post-session analysis.
    """

    def __init__(self, debounce=DEBOUNCE_FRAMES):
        self.debounce = debounce
        self._current_activity = "Unknown"
        self._candidate = "Unknown"
        self._candidate_count = 0
        self._timeline = []  # list of (activity, start_time)
        self._last_change_time = time.time()

    @property
    def current_activity(self):
        return self._current_activity

    @property
    def timeline(self):
        """Returns list of (activity, start_time, end_time) tuples."""
        result = []
        for i, (act, start) in enumerate(self._timeline):
            if i + 1 < len(self._timeline):
                end = self._timeline[i + 1][1]
            else:
                end = time.time()
            result.append((act, start, end))
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, landmarks):
        """
        Feed a frame of pose landmarks and return the stable activity label.

        Parameters
        ----------
        landmarks : dict
            Pose landmarks from PoseTracker.get_landmarks().
            Keys are MediaPipe pose landmark indices, values are (cx, cy, cz, vis).

        Returns
        -------
        str : stable activity label
        """
        if not landmarks:
            return self._current_activity

        raw = self._classify_raw(landmarks)
        return self._apply_debounce(raw)

    def get_timeline_summary(self):
        """Return a formatted string summarizing the activity timeline."""
        tl = self.timeline
        if not tl:
            return "No activity recorded."

        lines = []
        for act, start, end in tl:
            duration = end - start
            lines.append(f"  {act}: {duration:.1f}s")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Raw classification (frame-level, no smoothing)
    # ------------------------------------------------------------------

    def _classify_raw(self, lm):
        """
        Heuristic posture classification from pose landmarks.

        Uses relative positions of key body landmarks to determine posture.

        Landmark indices used:
            0  = Nose
            11 = Left Shoulder    12 = Right Shoulder
            13 = Left Elbow       14 = Right Elbow
            15 = Left Wrist       16 = Right Wrist
            23 = Left Hip         24 = Right Hip
            25 = Left Knee        26 = Right Knee
        """
        # Ensure we have the critical landmarks
        required = [0, 11, 12, 15, 16, 23, 24]
        for idx in required:
            if idx not in lm:
                return self._current_activity

        nose = lm[0]
        l_shoulder = lm[11]
        r_shoulder = lm[12]
        l_wrist = lm[15]
        r_wrist = lm[16]
        l_hip = lm[23]
        r_hip = lm[24]

        # Derived measurements
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        shoulder_x_center = (l_shoulder[0] + r_shoulder[0]) / 2
        hip_x_center = (l_hip[0] + r_hip[0]) / 2
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])

        # Torso height in pixels
        torso_height = abs(hip_y - shoulder_y)
        if torso_height < 10:
            torso_height = 10  # avoid division by zero

        # ---- T-Pose check ----
        # Both wrists roughly at shoulder height and wide apart
        wrist_at_shoulder_l = abs(l_wrist[1] - l_shoulder[1]) < torso_height * 0.25
        wrist_at_shoulder_r = abs(r_wrist[1] - r_shoulder[1]) < torso_height * 0.25
        wrists_wide = abs(l_wrist[0] - r_wrist[0]) > shoulder_width * 2.0
        if wrist_at_shoulder_l and wrist_at_shoulder_r and wrists_wide:
            return "T-Pose"

        # ---- Arms Raised check ----
        # Both wrists significantly above shoulders
        both_arms_up = (l_wrist[1] < l_shoulder[1] - torso_height * 0.3 and
                        r_wrist[1] < r_shoulder[1] - torso_height * 0.3)
        if both_arms_up:
            return "Arms Raised"

        # ---- Hands on Hips check ----
        # Both wrists near hip height and close to torso x-center
        wrist_near_hip_l = abs(l_wrist[1] - l_hip[1]) < torso_height * 0.3
        wrist_near_hip_r = abs(r_wrist[1] - r_hip[1]) < torso_height * 0.3
        wrist_close_l = abs(l_wrist[0] - l_hip[0]) < shoulder_width * 0.5
        wrist_close_r = abs(r_wrist[0] - r_hip[0]) < shoulder_width * 0.5
        if wrist_near_hip_l and wrist_near_hip_r and wrist_close_l and wrist_close_r:
            return "Hands on Hips"

        # ---- Leaning check ----
        # Shoulder center significantly offset from hip center
        lean_ratio = (shoulder_x_center - hip_x_center) / shoulder_width if shoulder_width > 0 else 0
        if lean_ratio > 0.35:
            return "Leaning Right"
        if lean_ratio < -0.35:
            return "Leaning Left"

        # ---- Sitting vs Standing ----
        # If knees are visible and roughly at hip level, likely sitting
        if 25 in lm and 26 in lm:
            l_knee = lm[25]
            r_knee = lm[26]
            knee_y = (l_knee[1] + r_knee[1]) / 2

            # Sitting: knees are close to hip level (small vertical gap)
            knee_hip_gap = abs(knee_y - hip_y) / torso_height
            if knee_hip_gap < 0.4:
                return "Sitting"

        return "Standing"

    # ------------------------------------------------------------------
    # Temporal debounce
    # ------------------------------------------------------------------

    def _apply_debounce(self, raw_label):
        """
        Apply N-frame debounce: only transition to a new state after
        seeing it for `self.debounce` consecutive frames.
        """
        if raw_label == self._current_activity:
            # Already in this state, reset candidate
            self._candidate = raw_label
            self._candidate_count = 0
            return self._current_activity

        if raw_label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw_label
            self._candidate_count = 1

        if self._candidate_count >= self.debounce:
            # Transition
            self._current_activity = raw_label
            self._candidate_count = 0
            now = time.time()
            self._timeline.append((raw_label, now))
            self._last_change_time = now

        return self._current_activity
