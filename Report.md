# Multi-Modal Human Pose and Gesture Estimation System v2.0
**Course:** Multimedia Analysis -- Advanced Project

---

## 1. Executive Summary

This project implements a **state-of-the-art multi-modal real-time human analysis system** that simultaneously processes body pose, hand gestures, facial expressions, exercise counting, posture symmetry, and full-body activity classification from a single webcam feed. It features motion trajectory visualization, live scrolling analytics graphs, background segmentation, and dual recording modes (CSV data + MP4 video with HUD overlay).

The system is built on a modular `src/` package architecture using the MediaPipe ecosystem and OpenCV 4, comprising **15 Python source files** across 6 packages.

### Capability Matrix

| Modality | Key Features |
|---|---|
| **Body Pose** | 33-landmark detection, 9 joint angles, velocity tracking, bounding box, visibility-weighted confidence |
| **Hand Gestures** | Dual-hand tracking, 12+ gesture vocabulary, finger curl ratios, hand openness %, pinch distance |
| **Face Mesh** | 468-landmark mesh, Eye Aspect Ratio (EAR) blink detection, Mouth Aspect Ratio (MAR), head tilt, expression classification |
| **Activity** | 7-posture classification (Standing, Sitting, T-Pose, Arms Raised, Leaning, Hands on Hips), 8-frame debounce state machine, timeline |
| **Exercise Counting** | Rep counter for bicep curls, squats, shoulder presses via angle-based state machine |
| **Symmetry Analysis** | Bilateral joint comparison (elbows, shoulders, knees, hips), per-pair and overall symmetry score |
| **Trajectory Trails** | Glowing fading motion trails for wrists and ankles (40-frame history) |
| **Live Graphs** | Real-time scrolling strip charts for elbow angles, EAR, MAR (120-frame window) |
| **Background Seg.** | MediaPipe segmentation mask with blur/color-gradient replacement modes |
| **Video Recording** | MP4 capture of analyzed video with full HUD overlay |
| **CSV Recording** | Per-frame data export with session summary statistics |
| **Visualization** | 7-panel semi-transparent HUD dashboard with color-coded analytics |

---

## 2. System Architecture

```
main.py                         # Orchestration + 11 keyboard controls
|
+-- src/
    +-- core/
    |   +-- pose.py              # PoseTracker      - BlazePose + kinematics + segmentation
    |   +-- hands.py             # GestureTracker   - Hand mesh + 12+ gesture classification
    |   +-- face.py              # FaceTracker      - FaceMesh + EAR/MAR/expression
    |   +-- activity.py          # ActivityClassifier - Posture state machine
    |   +-- exercise.py          # ExerciseCounter  - Rep counting (5 exercises)
    |   +-- symmetry.py          # SymmetryAnalyzer - Bilateral joint comparison
    |
    +-- utils/
        +-- angles.py            # 2D/3D angle math, distances, midpoint
        +-- smoothing.py         # Exponential Moving Average filter bank
        +-- recorder.py          # CSV SessionRecorder + MP4 VideoRecorder
        +-- visuals.py           # Multi-panel HUD dashboard renderer
        +-- trails.py            # Joint trajectory trail renderer
        +-- graphs.py            # Real-time scrolling line graph system
```

### Data Flow

```
Webcam Frame (1280x720)
    |
    +---> PoseTracker.process_frame()
    |         +---> get_full_pose_data() -> angles, velocities, bbox, confidence
    |         +---> apply_segmentation() -> blur/color background
    |         |
    |         +---> ActivityClassifier.update() -> posture label (debounced)
    |         +---> ExerciseCounter.update()    -> rep counts per exercise
    |         +---> SymmetryAnalyzer.analyze()  -> bilateral scores
    |         +---> TrailRenderer.update()      -> motion path buffer
    |
    +---> GestureTracker.process_hands()
    |         +---> get_full_hand_data() -> gesture, curl, openness, pinch
    |
    +---> FaceTracker.process_frame()
    |         +---> get_face_data() -> EAR, MAR, tilt, expression
    |
    +---> TrailRenderer.draw()     -> glowing trails overlay
    +---> GraphPanel.draw()        -> live scrolling graphs
    +---> Visualizer.draw_hud()    -> 7-panel analytics dashboard
    |
    +---> VideoRecorder.write_frame() -> MP4 output
    +---> SessionRecorder.record_frame() -> CSV buffer
    |
    +---> cv2.imshow()
```

---

## 3. Mathematical Foundations

### 3.1 Joint Angle Computation (2D Kinematics)
```python
angle = degrees(atan2(cy - by, cx - bx) - atan2(ay - by, ax - bx))
# Normalized to [0, 180] degrees (interior angle)
```
Applied to 9 joints: both elbows, shoulders, knees, hips, and neck.

### 3.2 3D Joint Angle (Vector Algebra)
```
cos(theta) = (BA . BC) / (|BA| * |BC|)
```
Uses dot-product for true spatial angle when 3D coordinates are available.

### 3.3 Eye Aspect Ratio (EAR) -- Blink Detection
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
6-point formula (Soukupova & Cech, 2016). Threshold: 0.21.

### 3.4 Mouth Aspect Ratio (MAR)
```
MAR = (V_center + V_left + V_right) / (2 * H)
```
Thresholds: >0.7 = Yawning, >0.35 = Talking.

### 3.5 Head Tilt (Roll Angle)
```
tilt = atan2(dy, dx)  // from eye center positions
```
Threshold: >12 degrees triggers Head Tilt classification.

### 3.6 Exponential Moving Average (EMA)
```
smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_(t-1)
```
Applied to all real-time metrics (alpha: 0.35 for angles, 0.4 for face ratios).

### 3.7 Temporal Debounce State Machine
Activity classifier requires 8 consecutive frames of same classification before transitioning state, preventing flickery labels.

### 3.8 Exercise Rep State Machine
Uses angle thresholds to detect up/down phases. A complete cycle (up -> down -> up) counts as one rep:
- Bicep curl: elbow <50 deg (down), >140 deg (up)
- Squat: knee <90 deg (down), >160 deg (up)
- Shoulder press: shoulder <50 deg (down), >140 deg (up)

### 3.9 Symmetry Scoring
```
symmetry = max(0, 1 - |L_angle - R_angle| / max(L_angle, R_angle)) * 100%
```
Overall score = average of all bilateral pair scores.

### 3.10 Finger Curl Ratio
```
curl = dist(tip, mcp) / (dist(wrist, mcp) * 1.8)  // clamped [0, 1]
```
0 = fully curled, 1 = fully extended. Hand openness = average curl ratio * 100%.

---

## 4. Gesture Recognition (12+ Classes)

| Gesture | Detection Logic |
|---|---|
| Fist | 0 fingers up |
| Pointing | Index finger only |
| Peace Sign | Index + Middle only |
| Rock / Metal | Index + Pinky (no thumb) |
| I Love You | Thumb + Index + Pinky |
| Call Me | Thumb + Pinky only |
| OK Sign | Thumb-Index pinch < 35% wrist-MCP distance |
| Gun | Thumb + Index only |
| Thumbs Up/Down | Thumb only, tip above/below IP joint |
| Open Hand | All 5 fingers |
| Three / Four | 3 or 4 specific fingers |

---

## 5. Activity Classification (7 Postures)

| Activity | Heuristic |
|---|---|
| T-Pose | Wrists at shoulder height + wide apart (>2x shoulder width) |
| Arms Raised | Both wrists >30% torso-height above shoulders |
| Hands on Hips | Wrists near hip height + close to torso |
| Leaning L/R | Shoulder center offset >35% of shoulder width from hips |
| Sitting | Knee-to-hip gap <40% torso height |
| Standing | Default |

---

## 6. Advanced Visualizations

### 6.1 Trajectory Trails
- Tracks 4 joints: L/R Wrist, L/R Ankle
- 40-frame circular buffer per joint
- Fading line segments: thickness and color intensity decrease with age
- Alpha-blended overlay for glow effect

### 6.2 Live Scrolling Graphs
- 4 parallel strip charts: R.Elbow angle, L.Elbow angle, EAR, MAR
- 120-frame scrolling window per graph
- Semi-transparent background with Y-axis labels and current value display
- Real-time data pushing each frame

### 6.3 Background Segmentation
- Uses MediaPipe's built-in segmentation mask (threshold: 0.5)
- Mode 1: Gaussian blur (55x55 kernel) on background
- Mode 2: Dark blue-purple gradient replacement on background
- Cycles through Off -> Blur -> Color -> Off with 'B' key

---

## 7. Recording & Export

### 7.1 CSV Session Recording
- Per-frame columns: all 9 joint angles, body confidence, L/R hand gestures, openness, pinch distance, expression, EAR L/R, MAR, head tilt, activity, exercise reps, symmetry score
- Session summary: total frames, duration, avg FPS, most common gesture/expression/activity, avg angles, total reps

### 7.2 Video Recording (MP4)
- Records the fully rendered frame including all HUD overlays
- MP4V codec at 20 FPS
- Captures exactly what the user sees on screen

---

## 8. Environment & Execution

### Prerequisites
- Python 3.11 or 3.12
- Webcam (built-in or external)

### Setup
```bash
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Execution
```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `P` | Toggle Pose analysis on/off |
| `H` | Toggle Hand analysis on/off |
| `F` | Toggle Face analysis on/off |
| `T` | Toggle Motion Trajectory Trails |
| `G` | Toggle Live Analytics Graphs |
| `B` | Cycle Background Segmentation (Off/Blur/Color) |
| `R` | Start/Stop CSV session recording |
| `V` | Start/Stop MP4 video recording (with HUD) |
| `S` | Save PNG snapshot |
| `X` | Reset exercise rep counters |
| `Q` | Quit and display session summary |

### Output Files
- **CSV Recordings**: `recordings/session_YYYYMMDD_HHMMSS.csv`
- **Video Recordings**: `recordings/video_YYYYMMDD_HHMMSS.mp4`
- **Snapshots**: `snapshots/snapshot_YYYYMMDD_HHMMSS.png`

---

## 9. HUD Dashboard Layout

Seven analytics panels on the right side:

1. **Pose Analytics** -- 9 joint angles + body confidence %
2. **Hand Analytics** -- Per-hand gesture, openness %, pinch distance
3. **Face Analytics** -- Expression, EAR, MAR, head tilt
4. **Exercise Tracker** -- Per-exercise rep count and current phase
5. **Symmetry Analysis** -- Overall score + per-pair breakdown (color-coded)
6. **Activity** -- Current posture classification

System bar (top-left):
- FPS counter with EMA smoothing
- CSV recording indicator (red dot)
- Video recording indicator (orange dot)
- Module toggle status for all 6 toggleable features
- Segmentation mode indicator
- Control key hints

All panels use semi-transparent backgrounds (55% opacity) with color-coded text.

---

## 10. Project Statistics

| Metric | Value |
|---|---|
| Total Python source files | 15 |
| Core analysis engines | 6 (pose, hands, face, activity, exercise, symmetry) |
| Utility modules | 6 (angles, smoothing, recorder, visuals, trails, graphs) |
| Tracked joint angles | 9 |
| Gesture vocabulary | 12+ |
| Activity classifications | 7 |
| Exercise types | 5 |
| Keyboard shortcuts | 11 |
| HUD panels | 7 |
| Recording modes | 3 (CSV, Video, Snapshot) |
| Real-time graphs | 4 |
| Trajectory-tracked joints | 4 |
| Mathematical formulas | 10 |
