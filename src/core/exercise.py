"""
Exercise Repetition Counter using joint angle state machine.

Tracks exercises like bicep curls, squats, and shoulder presses by
monitoring specific joint angles and detecting transitions between
'up' and 'down' phases. Each full cycle (up->down->up or down->up->down)
increments the rep counter.

Supports multiple simultaneous exercise tracking and provides per-exercise
statistics (reps, current phase, angle range).
"""

import time


# Exercise definitions: (name, joint_angle_key, down_threshold, up_threshold)
# When angle < down_threshold -> "down" phase
# When angle > up_threshold   -> "up" phase
EXERCISE_PROFILES = {
    'bicep_curl_right': {
        'angle_key': 'right_elbow',
        'down_threshold': 50,   # arm fully curled
        'up_threshold': 140,    # arm extended
        'label': 'R. Bicep Curl',
    },
    'bicep_curl_left': {
        'angle_key': 'left_elbow',
        'down_threshold': 50,
        'up_threshold': 140,
        'label': 'L. Bicep Curl',
    },
    'squat': {
        'angle_key': 'right_knee',  # track right knee for squats
        'down_threshold': 90,       # deep squat
        'up_threshold': 160,        # standing
        'label': 'Squat',
    },
    'shoulder_press_right': {
        'angle_key': 'right_shoulder',
        'down_threshold': 50,       # arm down
        'up_threshold': 140,        # arm overhead
        'label': 'R. Shoulder Press',
    },
    'shoulder_press_left': {
        'angle_key': 'left_shoulder',
        'down_threshold': 50,
        'up_threshold': 140,
        'label': 'L. Shoulder Press',
    },
}


class ExerciseCounter:
    """
    Multi-exercise repetition counter using angle-based state machine.

    For each exercise, tracks the current phase (up/down/transition) and
    counts completed rep cycles. A rep is counted when a full up->down->up
    cycle is completed.

    Usage:
        counter = ExerciseCounter()
        # each frame:
        counter.update(angles_dict)
        data = counter.get_exercise_data()
    """

    def __init__(self, exercises=None):
        """
        Parameters
        ----------
        exercises : list of str, optional
            Which exercises to track. Defaults to all available.
            Valid keys: 'bicep_curl_right', 'bicep_curl_left', 'squat',
                        'shoulder_press_right', 'shoulder_press_left'
        """
        if exercises is None:
            exercises = list(EXERCISE_PROFILES.keys())

        self._trackers = {}
        for name in exercises:
            if name in EXERCISE_PROFILES:
                profile = EXERCISE_PROFILES[name]
                self._trackers[name] = _RepTracker(
                    label=profile['label'],
                    angle_key=profile['angle_key'],
                    down_threshold=profile['down_threshold'],
                    up_threshold=profile['up_threshold'],
                )

    def update(self, angles, landmarks=None):
        """
        Feed a frame's worth of joint angles and landmarks into all exercise trackers.

        Parameters
        ----------
        angles : dict
            Joint angle dict from PoseTracker.compute_all_angles().
        landmarks : dict, optional
            Landmark dict from PoseTracker.get_landmarks().
        """
        for tracker in self._trackers.values():
            tracker.update(angles, landmarks)

    def get_exercise_data(self):
        """
        Return current state of all tracked exercises.

        Returns list of dicts:
        [{
            'label': str,
            'reps': int,
            'phase': str ('Up'/'Down'/'Transition'),
            'current_angle': float or None,
            'active': bool,
            'feedback': str,
        }]
        """
        data = []
        for tracker in self._trackers.values():
            data.append(tracker.get_data())
        return data

    def get_total_reps(self):
        """Sum of all reps across all exercises."""
        return sum(t.reps for t in self._trackers.values())

    def reset(self):
        """Reset all rep counters."""
        for tracker in self._trackers.values():
            tracker.reset()


class _RepTracker:
    """Internal per-exercise state machine."""

    PHASE_UNKNOWN = 'Idle'
    PHASE_UP = 'Up'
    PHASE_DOWN = 'Down'

    def __init__(self, label, angle_key, down_threshold, up_threshold):
        self.label = label
        self.angle_key = angle_key
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold

        self.reps = 0
        self.phase = self.PHASE_UNKNOWN
        self.current_angle = None
        self.feedback = ""
        self._went_down = False
        self._perfect_depth = False
        self._active = False
        self._last_active_time = 0

    def update(self, angles, landmarks=None):
        """Update with new angle and landmark data for this frame."""
        angle = angles.get(self.angle_key)

        if angle is None:
            self._active = False
            return

        self.current_angle = angle
        self._active = True
        self._last_active_time = time.time()

        # State machine transitions
        if angle < self.down_threshold:
            if self.phase != self.PHASE_DOWN:
                self.phase = self.PHASE_DOWN
                self._went_down = True
                self.feedback = "Descending..."
            
            # Special check for Squats: verify "Perfect Depth"
            if self.label == "Squat" and landmarks:
                # Compare hip y to knee y (larger y is lower in image)
                r_hip_y, r_knee_y = landmarks.get(24, (0,0))[1], landmarks.get(26, (0,0))[1]
                if r_hip_y > r_knee_y - 10: # Hip is roughly level with or below knee
                    self._perfect_depth = True
                    self.feedback = "GOOD DEPTH!"

        elif angle > self.up_threshold:
            if self.phase != self.PHASE_UP:
                self.phase = self.PHASE_UP
                if self._went_down:
                    # Completed a full rep cycle
                    if self.label == "Squat" and not self._perfect_depth:
                        self.feedback = "Go deeper next time!"
                    else:
                        self.reps += 1
                        self.feedback = "Rep Counted!"
                    
                    self._went_down = False
                    self._perfect_depth = False

    def get_data(self):
        return {
            'label': self.label,
            'reps': self.reps,
            'phase': self.phase,
            'current_angle': round(self.current_angle, 1) if self.current_angle else None,
            'active': self._active,
            'feedback': self.feedback,
        }

    def reset(self):
        self.reps = 0
        self.phase = self.PHASE_UNKNOWN
        self.current_angle = None
        self._went_down = False
