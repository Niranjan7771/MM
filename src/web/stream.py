"""
Video stream manager for the Flask web dashboard.

Manages a shared camera capture and analysis pipeline in a background
thread. Provides MJPEG frame generators for Flask streaming responses
and thread-safe access to current analytics data.

Supports multiple stream modes:
- 'full': All analysis engines active (pose/hands/face/activity/exercise/symmetry)
- 'sign': Sign language mode (hands only with letter detection)
- 'game': Game mode (pose only for body control)
"""

import cv2
import time
import threading
import numpy as np
import collections

from src.core.motion_predictor import MotionPredictorEngine

from src.core.pose import PoseTracker
from src.core.hands import GestureTracker
from src.core.face import FaceTracker
from src.core.activity import ActivityClassifier
from src.core.exercise import ExerciseCounter
from src.core.symmetry import SymmetryAnalyzer
from src.core.sign_language import SignLanguageClassifier
from src.utils.trails import TrailRenderer
from src.utils.visuals import Visualizer
from src.utils.graphs import GraphPanel
from src.utils.recorder import SessionRecorder, VideoRecorder


class StreamManager:
    """
    Singleton-like video stream and analysis manager.

    Runs the camera capture and analysis pipeline in a background thread.
    Multiple Flask routes can read from the same shared state.
    """

    def __init__(self):
        # Analysis engines
        self.pose_tracker = PoseTracker(model_complex=1, enable_seg=True)
        self.gesture_tracker = GestureTracker()
        self.face_tracker = FaceTracker()
        self.activity_classifier = ActivityClassifier()
        self.exercise_counter = ExerciseCounter()
        self.symmetry_analyzer = SymmetryAnalyzer()
        self.sign_classifier = SignLanguageClassifier()
        self.trail_renderer = TrailRenderer()
        self.visualizer = Visualizer()
        self.graph_panel = GraphPanel()
        self.csv_recorder = SessionRecorder(output_dir="recordings")
        self.video_recorder = VideoRecorder(output_dir="recordings")
        self.motion_predictor = MotionPredictorEngine(model_path='models/motion_lstm.pth')
        self._history = collections.deque(maxlen=20)
        
        # Track predictions to calculate accuracy. Stores T+5 predictions.
        self._pred_queue = collections.deque(maxlen=5)
        for _ in range(5):
            self._pred_queue.append(None)
        self._current_accuracy = 100.0  # Rolling accuracy percentage

        # Module toggles
        self.modules_active = {
            'pose': True,
            'hands': True,
            'face': True,
        }
        self.trails_on = True
        self.graphs_on = True

        # Current analytics data (thread-safe via lock)
        self._lock = threading.Lock()
        self._current_frame = None
        self._current_analytics = {}
        self._current_sign = {'letter': None, 'confidence': 0.0, 'sentence': ''}
        self._current_game_state = {'action': 'idle', 'lean': 0.0, 'arms_up': False, 'crouching': False}

        # Camera
        self._cap = None
        self._running = False
        self._thread = None
        self._frame_w = 1280
        self._frame_h = 720

    def start(self):
        """Start the background capture + analysis thread."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_h)

        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame_jpeg(self):
        """Get the latest analyzed frame as JPEG bytes."""
        with self._lock:
            if self._current_frame is None:
                if not self._running:
                    # Serve an explicit error screen to prevent the generator from hanging infinitely
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(blank, "CAMERA IN USE OR UNAVAILABLE.", (100, 350), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.putText(blank, "Please close other apps using the camera and restart.", (100, 420), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    return jpeg.tobytes()
                return None
            _, jpeg = cv2.imencode('.jpg', self._current_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes()

    def get_analytics(self):
        """Get the latest analytics data dict."""
        with self._lock:
            return dict(self._current_analytics)

    def get_sign_data(self):
        """Get the latest sign language data."""
        with self._lock:
            return dict(self._current_sign)

    def get_game_state(self):
        """Get the latest game control state."""
        with self._lock:
            return dict(self._current_game_state)

    def toggle_module(self, module):
        """Toggle a module on/off. Returns new state."""
        if module in self.modules_active:
            self.modules_active[module] = not self.modules_active[module]
            return self.modules_active[module]
        elif module == 'trails':
            self.trail_renderer.enabled = not self.trail_renderer.enabled
            if not self.trail_renderer.enabled:
                self.trail_renderer.clear()
            self.trails_on = self.trail_renderer.enabled
            return self.trails_on
        elif module == 'graphs':
            self.graph_panel.enabled = not self.graph_panel.enabled
            self.graphs_on = self.graph_panel.enabled
            return self.graphs_on
        elif module == 'seg':
            self.pose_tracker.cycle_seg_mode()
            return self.pose_tracker.seg_mode_label
        return None

    def reset_exercise(self):
        self.exercise_counter.reset()

    def clear_sign_sentence(self):
        self.sign_classifier.clear_sentence()

    def sign_backspace(self):
        self.sign_classifier.backspace()

    def sign_space(self):
        self.sign_classifier.add_space()

    def take_snapshot(self):
        """Save current frame as PNG. Returns path."""
        import os
        with self._lock:
            if self._current_frame is None:
                return None
            os.makedirs("snapshots", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join("snapshots", f"snapshot_{ts}.png")
            cv2.imwrite(path, self._current_frame)
            return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Main capture + analysis loop running in background thread."""
        while self._running:
            success, img = self._cap.read()
            if not success:
                time.sleep(0.01)
                continue

            img = cv2.flip(img, 1)

            # Analysis
            pose_data = {'landmarks': {}, 'angles': {}, 'velocities': {},
                         'bbox': None, 'confidence': 0.0}
            hand_data = []
            face_data = {}
            activity_label = self.activity_classifier.current_activity
            exercise_data = []
            symmetry_data = {}
            sign_letter = None
            sign_conf = 0.0

            try:
                # 1. Pose
                if self.modules_active['pose']:
                    img = self.pose_tracker.process_frame(img, draw=False)
                    pose_data = self.pose_tracker.get_full_pose_data(img, draw=False)
                    img = self.pose_tracker.apply_segmentation(img)
                    
                    landmarks = pose_data.get('landmarks', {})
                    posture = pose_data.get('posture', "Straight")
                    img = self.visualizer.draw_neon_skeleton(img, landmarks, posture)
                    
                    angles = pose_data.get('angles', {})
                    activity_label = self.activity_classifier.update(landmarks)
                    self.exercise_counter.update(angles, landmarks=landmarks)
                    exercise_data = self.exercise_counter.get_exercise_data()
                    symmetry_data = self.symmetry_analyzer.analyze(angles)
                    self.trail_renderer.update(landmarks)

                    # Motion Prediction
                    ordered_keys = [
                        'left_elbow', 'right_elbow',
                        'left_shoulder', 'right_shoulder',
                        'left_hip', 'right_hip',
                        'left_knee', 'right_knee',
                        'neck_inclination'
                    ]
                    feature_vec = [angles.get(k, 0.0) or 0.0 for k in ordered_keys]
                    
                    # Evaluate accuracy of the prediction made 5 frames ago for THIS frame
                    past_pred = self._pred_queue.popleft()
                    if past_pred is not None:
                        # Calculate Mean Absolute Error across all 9 angles
                        mae = sum(abs(a - p) for a, p in zip(feature_vec, past_pred)) / len(feature_vec)
                        # An MAE of 0 is 100%, an MAE of 45+ is 0%
                        acc = max(0.0, 100.0 - (mae / 45.0 * 100.0))
                        # Smooth the accuracy
                        self._current_accuracy = self._current_accuracy * 0.9 + acc * 0.1
                    else:
                        self._pred_queue.append(None) 
                    
                    self._history.append(feature_vec)
                    prediction = self.motion_predictor.predict(list(self._history))
                    
                    if prediction is not None and len(prediction) == 5:
                        self._pred_queue.append(prediction[-1]) 
                    else:
                        self._pred_queue.append(None)
                        
                    pose_data['motion_prediction'] = prediction
                    pose_data['prediction_accuracy'] = self._current_accuracy

                    # Game state
                    self._update_game_state(landmarks, angles)

                # 2. Hands
                if self.modules_active['hands']:
                    img = self.gesture_tracker.process_hands(img, draw=False)
                    hand_data = self.gesture_tracker.get_full_hand_data(img)
                    
                    # Draw neon hands
                    for hd in hand_data:
                        lms = hd.get('landmarks_raw', {})
                        for lid, pos in lms.items():
                            cv2.circle(img, pos[:2], 3, (255, 255, 100), 1, cv2.LINE_AA)
                            cv2.circle(img, pos[:2], 1, (255, 255, 255), -1, cv2.LINE_AA)

                    # Sign language detection
                    if hand_data:
                        hd = hand_data[0]
                        sign_letter, sign_conf = self.sign_classifier.classify(
                            hd.get('landmarks_raw', hd.get('landmarks', {})),
                            hd.get('type', 'Right'),
                        )

                # 3. Face
                if self.modules_active['face']:
                    img = self.face_tracker.process_frame(img, draw=True)
                    face_data = self.face_tracker.get_face_data(img)

                # Visual layers
                img = self.trail_renderer.draw(img)
                self.graph_panel.update(pose_data.get('angles', {}), face_data)
                self.graph_panel.draw(img, 10, max(0, self._frame_h - 345))

                # Build analytics dict
                analytics = {
                    'pose': pose_data,
                    'hands': hand_data,
                    'face': face_data,
                    'activity': activity_label,
                    'exercise': exercise_data,
                    'symmetry': symmetry_data
                }

                # Thread-safe update
                with self._lock:
                    self._current_frame = img.copy()
                    self._current_analytics = analytics
                    self._current_sign = {
                        'letter': sign_letter,
                        'confidence': round(sign_conf, 2) if sign_conf else 0.0,
                        'sentence': self.sign_classifier.sentence,
                    }

            except Exception as e:
                print(f"CRITICAL ERROR IN CAPTURE LOOP: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
                continue

            # Small sleep to prevent CPU overload
            time.sleep(0.01)

    def _update_game_state(self, landmarks, angles):
        """Compute game control state from pose data."""
        if not landmarks:
            return

        required = [0, 11, 12, 15, 16, 23, 24]
        for idx in required:
            if idx not in landmarks:
                return

        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        shoulder_cx = (l_shoulder[0] + r_shoulder[0]) / 2
        hip_cx = (l_hip[0] + r_hip[0]) / 2
        shoulder_w = abs(r_shoulder[0] - l_shoulder[0])
        torso_h = abs((l_hip[1] + r_hip[1]) / 2 - (l_shoulder[1] + r_shoulder[1]) / 2)

        lean = 0.0
        if shoulder_w > 0:
            lean = (shoulder_cx - hip_cx) / shoulder_w

        arms_up = (l_wrist[1] < l_shoulder[1] - torso_h * 0.3 and
                   r_wrist[1] < r_shoulder[1] - torso_h * 0.3)

        crouching = False
        if 25 in landmarks and 26 in landmarks:
            knee_y = (landmarks[25][1] + landmarks[26][1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            crouching = abs(knee_y - hip_y) / max(torso_h, 1) < 0.4

        action = 'idle'
        if arms_up:
            action = 'jump'
        elif crouching:
            action = 'duck'
        elif lean > 0.3:
            action = 'right'
        elif lean < -0.3:
            action = 'left'

        with self._lock:
            self._current_game_state = {
                'action': action,
                'lean': round(lean, 2),
                'arms_up': arms_up,
                'crouching': crouching,
            }

    def _build_analytics(self, pose_data, hand_data, face_data,
                         activity_label, exercise_data, symmetry_data):
        """Build the JSON-ready analytics dict."""
        analytics = {
            'pose': {
                'confidence': pose_data.get('confidence', 0),
                'angles': {k: v for k, v in pose_data.get('angles', {}).items() if v is not None},
                'seg_mode': self.pose_tracker.seg_mode_label,
                'motion_prediction': pose_data.get('motion_prediction'),
                'prediction_accuracy': pose_data.get('prediction_accuracy', 0.0)
            },
            'hands': [],
            'face': face_data,
            'activity': activity_label,
            'exercise': exercise_data,
            'symmetry': symmetry_data,
            'modules': dict(self.modules_active),
            'trails': self.trails_on,
            'graphs': self.graphs_on,
        }

        for hd in hand_data:
            analytics['hands'].append({
                'type': hd.get('type', ''),
                'gesture': hd.get('gesture', ''),
                'openness': hd.get('openness', 0),
                'pinch_distance': hd.get('pinch_distance', 0),
            })

        return analytics
