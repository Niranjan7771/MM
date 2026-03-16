"""
Joint trajectory trail renderer.

Draws glowing motion trails showing the path of key joints (wrists, ankles,
index fingers) over the last N frames. Trails fade from bright to dim over
time, creating a visually striking motion visualization effect.
"""

import cv2
import numpy as np
from collections import deque


# Tracked joint configurations: (landmark_id, color_BGR, label)
# These use MediaPipe Pose landmark indices
POSE_TRAIL_JOINTS = [
    (16, (0, 200, 255), 'R.Wrist'),   # Right wrist - orange
    (15, (255, 200, 0), 'L.Wrist'),   # Left wrist  - cyan
    (28, (0, 100, 255), 'R.Ankle'),   # Right ankle  - red-orange
    (27, (255, 100, 0), 'L.Ankle'),   # Left ankle   - blue
]

# Max trail length (number of frames to keep)
TRAIL_LENGTH = 40

# Trail rendering
TRAIL_MAX_THICKNESS = 4
TRAIL_MIN_THICKNESS = 1


class TrailRenderer:
    """
    Renders fading motion trails for tracked body joints.

    Stores a circular buffer of positions per joint. On each draw call,
    renders connected line segments with decreasing thickness and opacity
    from newest to oldest point.
    """

    def __init__(self, trail_length=TRAIL_LENGTH):
        self.trail_length = trail_length
        self._trails = {}
        self._enabled = True

        # Initialize trail buffers for each tracked joint
        for lm_id, color, label in POSE_TRAIL_JOINTS:
            self._trails[lm_id] = {
                'points': deque(maxlen=trail_length),
                'color': color,
                'label': label,
            }

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    def update(self, pose_landmarks):
        """
        Record current joint positions from pose landmarks.

        Parameters
        ----------
        pose_landmarks : dict
            Landmarks from PoseTracker.get_landmarks().
            Keys are int IDs, values are (cx, cy, cz, visibility).
        """
        for lm_id in self._trails:
            if lm_id in pose_landmarks:
                x, y = pose_landmarks[lm_id][:2]
                vis = pose_landmarks[lm_id][3] if len(pose_landmarks[lm_id]) > 3 else 1.0
                if vis > 0.5:  # Only record if landmark is visible
                    self._trails[lm_id]['points'].append((x, y))
            # If landmark not visible, don't append (creates gap in trail)

    def draw(self, img):
        """
        Render all trails onto the image.

        Uses addWeighted blending for a glowing effect. Lines fade from
        full brightness to dim as they age.
        """
        if not self._enabled:
            return img

        overlay = img.copy()

        for lm_id, trail_data in self._trails.items():
            points = trail_data['points']
            color = trail_data['color']

            if len(points) < 2:
                continue

            pts = list(points)
            n = len(pts)

            for i in range(1, n):
                # Fade factor: 0.0 (oldest) to 1.0 (newest)
                fade = i / n

                # Thickness decreases with age
                thickness = int(TRAIL_MIN_THICKNESS +
                              (TRAIL_MAX_THICKNESS - TRAIL_MIN_THICKNESS) * fade)

                # Color fades
                c = tuple(int(ch * fade) for ch in color)

                cv2.line(overlay, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)

            # Draw a bright dot at the current position
            cv2.circle(overlay, pts[-1], 5, color, cv2.FILLED, cv2.LINE_AA)

        # Blend the trails onto the original image
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        return img

    def clear(self):
        """Clear all trail histories."""
        for trail_data in self._trails.values():
            trail_data['points'].clear()
