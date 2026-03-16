"""
Session data recorder for analytics export.

Records per-frame analytics (joint angles, gestures, expressions, activity)
to an in-memory buffer during a live session, then exports to CSV on demand.
Also computes session summary statistics.

Includes VideoRecorder for capturing the analyzed video feed with HUD
overlays as an MP4 file.
"""

import csv
import cv2
import os
import time
from collections import Counter


class VideoRecorder:
    """
    Records video frames with all HUD overlays to an MP4 file.

    Usage:
        vr = VideoRecorder()
        vr.start(width=1280, height=720)
        # each frame:
        vr.write_frame(img_with_hud)
        # done:
        path = vr.stop()
    """

    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self._writer = None
        self._filepath = None
        self._recording = False

    @property
    def is_recording(self):
        return self._recording

    def start(self, width, height, fps=20.0):
        """Begin recording to a new MP4 file."""
        os.makedirs(self.output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self.output_dir, f"video_{ts}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self._filepath, fourcc, fps, (width, height))
        self._recording = True

    def write_frame(self, img):
        """Write a single frame to the video file."""
        if self._recording and self._writer is not None:
            self._writer.write(img)

    def stop(self):
        """Stop recording and finalize the video file. Returns filepath."""
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return os.path.abspath(self._filepath) if self._filepath else None

class SessionRecorder:
    """
    Buffers per-frame analysis data and exports to CSV.

    Usage:
        recorder = SessionRecorder()
        recorder.start()
        # ... each frame:
        recorder.record_frame(frame_data_dict)
        # ... when done:
        filepath = recorder.stop_and_export()
    """

    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self._buffer = []
        self._recording = False
        self._start_time = None
        self._frame_count = 0

    @property
    def is_recording(self):
        return self._recording

    def start(self):
        """Begin a new recording session."""
        self._buffer = []
        self._recording = True
        self._start_time = time.time()
        self._frame_count = 0

    def record_frame(self, data):
        """
        Record a single frame of analytics data.

        Parameters
        ----------
        data : dict
            Expected keys (all optional, missing values become ''):
            - right_elbow_angle, left_elbow_angle
            - right_knee_angle, left_knee_angle
            - right_shoulder_angle, left_shoulder_angle
            - body_confidence
            - gesture_left, gesture_right
            - hand_openness_left, hand_openness_right
            - pinch_distance_left, pinch_distance_right
            - expression
            - ear_left, ear_right, mar
            - head_tilt
            - activity
        """
        if not self._recording:
            return

        self._frame_count += 1
        row = {
            'frame': self._frame_count,
            'timestamp': round(time.time() - self._start_time, 3),
        }
        row.update(data)
        self._buffer.append(row)

    def stop_and_export(self):
        """
        Stop recording and export the buffered data to a timestamped CSV file.

        Returns
        -------
        str : absolute path to the exported CSV file, or None if no data
        """
        self._recording = False

        if not self._buffer:
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp_str}.csv"
        filepath = os.path.join(self.output_dir, filename)

        # Collect all unique keys across all rows for the header
        all_keys = []
        seen = set()
        for row in self._buffer:
            for key in row:
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for row in self._buffer:
                writer.writerow(row)

        return os.path.abspath(filepath)

    def get_summary(self):
        """
        Compute session summary statistics.

        Returns
        -------
        dict with keys:
            total_frames, duration_seconds, avg_fps,
            most_common_gesture_left, most_common_gesture_right,
            most_common_expression, most_common_activity,
            avg_right_elbow_angle, avg_left_elbow_angle
        """
        if not self._buffer:
            return {'total_frames': 0}

        duration = self._buffer[-1].get('timestamp', 0)
        total = len(self._buffer)

        summary = {
            'total_frames': total,
            'duration_seconds': round(duration, 2),
            'avg_fps': round(total / duration, 1) if duration > 0 else 0,
        }

        # Most common categorical values
        for field, label in [
            ('gesture_left', 'most_common_gesture_left'),
            ('gesture_right', 'most_common_gesture_right'),
            ('expression', 'most_common_expression'),
            ('activity', 'most_common_activity'),
        ]:
            values = [r.get(field) for r in self._buffer if r.get(field)]
            if values:
                summary[label] = Counter(values).most_common(1)[0][0]
            else:
                summary[label] = 'N/A'

        # Average numeric values
        for field in ['right_elbow_angle', 'left_elbow_angle',
                      'right_knee_angle', 'left_knee_angle']:
            values = []
            for r in self._buffer:
                v = r.get(field)
                if v is not None and v != '':
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if values:
                summary[f'avg_{field}'] = round(sum(values) / len(values), 1)

        return summary
