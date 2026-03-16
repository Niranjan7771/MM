"""
Multi-Modal Human Pose & Gesture Estimation System
===================================================

Advanced real-time analysis pipeline combining:
- Full-body pose estimation with multi-joint kinematics
- Dual-hand gesture recognition with finger analytics
- Face mesh expression estimation (EAR/MAR/head tilt)
- Activity/posture classification with temporal smoothing
- Exercise repetition counting (bicep curls, squats, shoulder press)
- Bilateral pose symmetry analysis
- Motion trajectory trails with glow effect
- Real-time scrolling analytics graphs
- Background segmentation (blur / color replace)
- Video recording with HUD overlay
- CSV session data export
- Premium multi-panel HUD dashboard

Keyboard Controls:
    P  - Toggle pose analysis
    H  - Toggle hand analysis
    F  - Toggle face analysis
    T  - Toggle motion trails
    G  - Toggle live graphs
    B  - Cycle background segmentation (Off / Blur / Color)
    R  - Start / stop CSV session recording
    V  - Start / stop video recording (MP4 with HUD)
    S  - Save PNG snapshot
    X  - Reset exercise rep counters
    Q  - Quit and show session summary
"""

import cv2
import os
import sys
import time

from src.core.pose import PoseTracker
from src.core.hands import GestureTracker
from src.core.face import FaceTracker
from src.core.activity import ActivityClassifier
from src.core.exercise import ExerciseCounter
from src.core.symmetry import SymmetryAnalyzer
from src.utils.visuals import Visualizer
from src.utils.trails import TrailRenderer
from src.utils.graphs import GraphPanel
from src.utils.recorder import SessionRecorder, VideoRecorder


def main():
    print("=" * 62)
    print("  Multi-Modal Human Pose & Gesture Estimation System")
    print("  Advanced Real-Time Analysis Pipeline v2.0")
    print("=" * 62)
    print()

    # ----- Initialize Engines -----
    print("[INIT] Pose Tracker (BlazePose + Segmentation)...")
    pose_tracker = PoseTracker(model_complex=1, enable_seg=True)

    print("[INIT] Gesture Tracker (MediaPipe Hands)...")
    gesture_tracker = GestureTracker()

    print("[INIT] Face Tracker (FaceMesh 468)...")
    face_tracker = FaceTracker()

    print("[INIT] Activity Classifier (State Machine)...")
    activity_classifier = ActivityClassifier()

    print("[INIT] Exercise Counter (5 exercises)...")
    exercise_counter = ExerciseCounter()

    print("[INIT] Symmetry Analyzer...")
    symmetry_analyzer = SymmetryAnalyzer()

    print("[INIT] Visualizer (HUD Dashboard)...")
    visualizer = Visualizer()

    print("[INIT] Trail Renderer (Trajectory Visualization)...")
    trail_renderer = TrailRenderer()

    print("[INIT] Graph Panel (Live Analytics)...")
    graph_panel = GraphPanel()

    print("[INIT] Session Recorder (CSV)...")
    csv_recorder = SessionRecorder(output_dir="recordings")

    print("[INIT] Video Recorder (MP4)...")
    video_recorder = VideoRecorder(output_dir="recordings")

    # ----- Module toggle state -----
    modules_active = {
        'pose': True,
        'hands': True,
        'face': True,
    }

    # ----- Open Video Source -----
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open video source (webcam 0).")
        sys.exit(1)

    # Get actual frame dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print()
    print("[READY] Pipeline initialized. Controls:")
    print("   P = Pose      H = Hands     F = Face")
    print("   T = Trails    G = Graphs    B = Bg Seg (cycle)")
    print("   R = CSV Rec   V = Video Rec S = Snapshot")
    print("   X = Reset Reps              Q = Quit")
    print()

    snapshot_dir = "snapshots"

    # ----- Main Loop -----
    while True:
        success, img = cap.read()
        if not success:
            continue

        # Mirror for natural selfie-view
        img = cv2.flip(img, 1)

        # ---- Analysis Pipeline ----
        pose_data = {'landmarks': {}, 'angles': {}, 'velocities': {},
                     'bbox': None, 'confidence': 0.0}
        hand_data = []
        face_data = {}
        activity_label = activity_classifier.current_activity
        exercise_data = []
        symmetry_data = {}

        # 1. Pose
        if modules_active['pose']:
            img = pose_tracker.process_frame(img, draw=True)
            pose_data = pose_tracker.get_full_pose_data(img, draw=True)

            # Background segmentation
            img = pose_tracker.apply_segmentation(img)

            # Feed pose into sub-analyzers
            landmarks = pose_data.get('landmarks', {})
            angles = pose_data.get('angles', {})

            activity_label = activity_classifier.update(landmarks)
            exercise_counter.update(angles)
            exercise_data = exercise_counter.get_exercise_data()
            symmetry_data = symmetry_analyzer.analyze(angles)

            # Update trajectory trails
            trail_renderer.update(landmarks)

        # 2. Hands
        if modules_active['hands']:
            img = gesture_tracker.process_hands(img, draw=True)
            hand_data = gesture_tracker.get_full_hand_data(img)

        # 3. Face
        if modules_active['face']:
            img = face_tracker.process_frame(img, draw=True)
            face_data = face_tracker.get_face_data(img)

        # ---- Visual Layers ----

        # Draw trajectory trails
        img = trail_renderer.draw(img)

        # Draw live graphs (bottom-left)
        graph_panel.update(pose_data.get('angles', {}), face_data)
        graph_panel.draw(img, GRAPH_X, GRAPH_Y(frame_h))

        # ---- HUD Dashboard ----
        img = visualizer.draw_hud(
            img, pose_data, hand_data, face_data,
            activity_label, exercise_data, symmetry_data,
            modules_active,
            csv_recorder.is_recording, video_recorder.is_recording,
            pose_tracker.seg_mode_label,
            trail_renderer.enabled, graph_panel.enabled,
        )

        # ---- Recording ----
        if csv_recorder.is_recording:
            frame_record = _build_frame_record(
                pose_data, hand_data, face_data, activity_label,
                exercise_data, symmetry_data,
            )
            csv_recorder.record_frame(frame_record)

        if video_recorder.is_recording:
            video_recorder.write_frame(img)

        # ---- Display ----
        cv2.imshow("Multi-Modal Estimation - Advanced v2.0", img)

        # ---- Keyboard Input ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            modules_active['pose'] = not modules_active['pose']
            _log_toggle("Pose", modules_active['pose'])

        elif key == ord('h'):
            modules_active['hands'] = not modules_active['hands']
            _log_toggle("Hands", modules_active['hands'])

        elif key == ord('f'):
            modules_active['face'] = not modules_active['face']
            _log_toggle("Face", modules_active['face'])

        elif key == ord('t'):
            trail_renderer.enabled = not trail_renderer.enabled
            _log_toggle("Trails", trail_renderer.enabled)
            if not trail_renderer.enabled:
                trail_renderer.clear()

        elif key == ord('g'):
            graph_panel.enabled = not graph_panel.enabled
            _log_toggle("Graphs", graph_panel.enabled)

        elif key == ord('b'):
            pose_tracker.cycle_seg_mode()
            print(f"[SEG] Background: {pose_tracker.seg_mode_label}")

        elif key == ord('r'):
            if csv_recorder.is_recording:
                filepath = csv_recorder.stop_and_export()
                if filepath:
                    print(f"[CSV] Recording saved: {filepath}")
                else:
                    print("[CSV] Recording stopped (no data).")
            else:
                csv_recorder.start()
                print("[CSV] Recording started...")

        elif key == ord('v'):
            if video_recorder.is_recording:
                filepath = video_recorder.stop()
                if filepath:
                    print(f"[VID] Video saved: {filepath}")
            else:
                video_recorder.start(frame_w, frame_h, fps=20.0)
                print("[VID] Video recording started...")

        elif key == ord('s'):
            os.makedirs(snapshot_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(snapshot_dir, f"snapshot_{ts}.png")
            cv2.imwrite(snap_path, img)
            print(f"[SNAP] Saved: {snap_path}")

        elif key == ord('x'):
            exercise_counter.reset()
            print("[RESET] Exercise counters reset.")

    # ----- Cleanup -----
    cap.release()
    cv2.destroyAllWindows()

    # Auto-stop recorders
    if csv_recorder.is_recording:
        filepath = csv_recorder.stop_and_export()
        if filepath:
            print(f"[CSV] Auto-saved: {filepath}")

    if video_recorder.is_recording:
        filepath = video_recorder.stop()
        if filepath:
            print(f"[VID] Auto-saved: {filepath}")

    # Session summary
    summary = csv_recorder.get_summary()
    if summary.get('total_frames', 0) > 0:
        print()
        print("=" * 54)
        print("  SESSION SUMMARY")
        print("=" * 54)
        for k, v in summary.items():
            print(f"  {k.replace('_', ' ').title()}: {v}")
        print(f"  Total Exercise Reps: {exercise_counter.get_total_reps()}")
        print("=" * 54)

    # Activity timeline
    timeline = activity_classifier.get_timeline_summary()
    if timeline and timeline != "No activity recorded.":
        print()
        print("Activity Timeline:")
        print(timeline)

    print()
    print("[EXIT] System shut down cleanly.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Graph positioning (bottom-left)
GRAPH_X = 10

def GRAPH_Y(frame_h):
    return frame_h - 345  # stack 4 graphs above bottom


def _log_toggle(name, state):
    s = "ON" if state else "OFF"
    print(f"[TOGGLE] {name}: {s}")


def _build_frame_record(pose_data, hand_data, face_data,
                        activity_label, exercise_data, symmetry_data):
    """Flatten all analytics into a single dict for CSV recording."""
    record = {}

    # Pose angles
    for name, val in pose_data.get('angles', {}).items():
        record[name + '_angle'] = val if val is not None else ''
    record['body_confidence'] = pose_data.get('confidence', '')

    # Hands
    gesture_l = gesture_r = ''
    openness_l = openness_r = ''
    pinch_l = pinch_r = ''
    for hd in hand_data:
        if hd['type'] == 'Left':
            gesture_l = hd.get('gesture', '')
            openness_l = hd.get('openness', '')
            pinch_l = hd.get('pinch_distance', '')
        elif hd['type'] == 'Right':
            gesture_r = hd.get('gesture', '')
            openness_r = hd.get('openness', '')
            pinch_r = hd.get('pinch_distance', '')

    record['gesture_left'] = gesture_l
    record['gesture_right'] = gesture_r
    record['hand_openness_left'] = openness_l
    record['hand_openness_right'] = openness_r
    record['pinch_distance_left'] = pinch_l
    record['pinch_distance_right'] = pinch_r

    # Face
    record['expression'] = face_data.get('expression', '')
    record['ear_left'] = face_data.get('ear_left', '')
    record['ear_right'] = face_data.get('ear_right', '')
    record['mar'] = face_data.get('mar', '')
    record['head_tilt'] = face_data.get('head_tilt', '')

    # Activity
    record['activity'] = activity_label

    # Exercise
    for ex in exercise_data:
        key = ex['label'].replace(' ', '_').replace('.', '').lower()
        record[f'reps_{key}'] = ex['reps']

    # Symmetry
    record['symmetry_overall'] = symmetry_data.get('overall_score', '')

    return record


if __name__ == "__main__":
    main()
