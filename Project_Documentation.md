# Multi-Modal Human Pose & Gesture Estimation System

Welcome to the Multi-Modal Human Pose & Gesture Estimation System! This comprehensive computer vision project provides a complete end-to-end pipeline for real-time human analysis from a standard webcam. It simultaneously tracks body pose, hand gestures, facial expressions, sign language, and maps body movements to a gesture-controlled game.

This document serves as the complete guide for anyone looking to understand, test, or modify the project.

---

## Quick Start: How to Test and Run

You can run this project in two different modes: a local desktop application (OpenCV visualizer) or a web-based dashboard (Flask + HTML/JS).

### Prerequisites
1. **Python 3.8+** installed.
2. A working **Webcam**.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Requires: `opencv-python`, `mediapipe`, `numpy`, `flask`)*

### Mode 1: Web Dashboard (Recommended)
This mode launches a local web server with 4 interactive pages (Home, Camera Analysis, Sign Language, Gesture Game). This was built to provide a modern, highly interactive UI separated from the raw computer vision pipeline.

**To run:**
```bash
python app.py
```
Then open your browser and navigate to: **http://localhost:5000**

### Mode 2: Standalone OpenCV Window
This mode is best for debugging the raw pipeline. It opens a standard window rendering the camera feed with a Heads-Up Display (HUD) overlay containing all the live statistics.

**To run:**
```bash
python main.py
```

**Keyboard Controls for Standalone Mode:**
- `P`: Toggle Pose Estimation
- `H`: Toggle Hand Tracking
- `F`: Toggle Face Mesh
- `T`: Toggle Motion Trails
- `G`: Toggle Scrolling Graphs
- `B`: Cycle Background Segmentation modes (None, Blur, Virtual Green Screen)
- `R`: Toggle CSV Recording
- `V`: Toggle MP4 Video Recording
- `S`: Take a quick Snapshot
- `X`: Reset Exercise Counters
- `Q`: Quit

---

## Project Architecture & Components

The project is highly modular, designed so that new "engines" can be added without modifying the core capture loops.

### Directory Structure
```
MM/
│
├── app.py                  # Entry point for the Web Dashboard (Flask)
├── main.py                 # Entry point for Standalone OpenCV Mode
├── requirements.txt        # Python dependencies
│
├── src/                    # Source code directory
│   ├── core/               # Core computer vision analysis engines
│   │   ├── activity.py     # Classifies full-body postures (T-Pose, Sitting, etc.)
│   │   ├── exercise.py     # Counts exercise reps (Squats, Curls, etc.)
│   │   ├── face.py         # Face landmarks, Blinks (EAR), Yawning (MAR)
│   │   ├── hands.py        # Hand landmark detection and static gesture classification
│   │   ├── pose.py         # Main body pose tracking and joint angle computation
│   │   ├── sign_language.py# ASL alphabet recognition
│   │   └── symmetry.py     # Analyzes bilateral symmetry of left/right limbs
│   │
│   ├── utils/              # Math, rendering, and recording utilities
│   │   ├── angles.py       # Math functions for 2D/3D joint angles
│   │   ├── graphs.py       # Renders real-time strip charts
│   │   ├── recorder.py     # Handles saving to CSV and MP4
│   │   ├── smoothing.py    # Exponential Moving Average (EMA) filters
│   │   ├── trails.py       # Renders motion trajectory trails
│   │   └── visuals.py      # Combines data into the Standalone OpenCV HUD
│   │
│   └── web/                # Backend Flask code
│       ├── routes.py       # Flask endpoints (API + HTML routing)
│       └── stream.py       # Thread-safe background stream manager
│
├── static/                 # Frontend Web Assets
│   ├── css/style.css       # Custom UI styling (Dark Theme)
│   └── js/                 # Client-side scripts polling the backend APIs
│       ├── camera.js       # Live analytics dashboard logic
│       ├── game.js         # The HTML5 Canvas dodging game
│       └── sign_language.js# UI logic for sentence building
│
├── templates/              # HTML files for Flask
│   ├── index.html          # Dashboard Home
│   ├── camera.html         # Camera Analysis view
│   ├── game.html           # Gesture Game view
│   └── sign_language.html  # Sign Language view
│
├── report/                 # LaTeX academic report detailing the math/formulas
└── screenshots/            # Examples of the Web UI used in the document
```

---

## Detailed Code Walkthrough & Design Decisions

Here is a breakdown of the most critical modules, explaining *how* they work and *why* specific design decisions were made.

### 1. `src/core/pose.py` (The Foundation)
**What it does:** Uses `mediapipe.solutions.pose` to find 33 body landmarks. It mathematically calculates the angles of 9 major joints (Left/Right Elbows, Shoulders, Knees, Hips, and Neck). It also handles background segmentation to blur out the room behind you.
**Design Decision - EMA Smoothing:** Raw landmark points from webcams naturally "jitter" frame-to-frame. If we calculate an angle directly from these jittery points, the resulting angle value bounces wildly, breaking downstream logic like exercise counting. I implemented an **Exponential Moving Average (EMA)** filter (`src/utils/smoothing.py`). It blends 35% of the *new* frame's value with 65% of the *previous* frame's value. This results in silky-smooth joint angle tracking.

### 2. `src/core/hands.py` (Hand Gestures)
**What it does:** Tracks 21 landmarks per hand. Instead of using a complex Machine Learning classifier, it classifies 12+ gestures strictly through *geometry* and *finger states*.
**Design Decision - Heuristics over ML:** I decided to determine finger states (is it curled or extended) by comparing the distance of the finger tip relative to the finger base joints. If you hold up just your index and middle fingers, it geometrically classifies this as a "Peace" sign. This deterministic, rule-based approach uses virtually zero CPU overhead compared to repeatedly running a CNN model, allowing the system to easily maintain high FPS.

### 3. `src/core/face.py` (Facial Expressions)
**What it does:** Tracks 468 points on the face. It specializes in two custom metrics:
*   **EAR (Eye Aspect Ratio):** To detect blinks. It measures the ratio of vertical eye openness to horizontal eye width. When EAR drops below `0.21`, you just squeezed your eye shut.
*   **MAR (Mouth Aspect Ratio):** To detect yawning or talking. Measures vertical mouth openness vs horizontal width.
**Design Decision:** Using specific landmark ratios (like EAR/MAR originally described in academic papers by Soukupová and Čech) allows the system to be distance-invariant. Whether you are 2 feet from the camera or 10 feet away, a ratio stays exactly the same, ensuring consistent blink detection without requiring calibration.

### 4. `src/core/activity.py` & `src/core/exercise.py` (Higher-Level Logic)
**What it does:** 
*   **Activity** checks the angles returning from `pose.py` and maps them to states (e.g., if wrists are roughly at shoulder height, wide apart, the user is in a "T-Pose").
*   **Exercise** counts reps (e.g., Squats). It works as a mathematically-bound state machine. E.g., for a Squat, if the knee interior angle drops below 90°, state transitions to `DOWN`. When it goes back above 160°, state goes to `UP` and `reps += 1`.
**Design Decision - Temporal Debounce:** In `activity.py`, human movement causes boundary transitions (e.g., smoothly raising arms triggers "Standing", then "T-Pose", then "Arms Raised"). To prevent the text UI from rapidly flashing between these states, I added an 8-frame "debounce". A posture MUST be held steadily for 8 consecutive frames before the system actually commits classifying it.

### 5. `src/core/sign_language.py` (ASL)
**What it does:** Similar to hands.py, but strictly focused on interpreting 10 static American Sign Language letters (A, B, C, D, I, L, O, V, W, Y). 
**Design Decision - Sentence Building Logic:** The web UI will string detected characters into a sentence. To avoid flooding the sentence with "AAAAAA", I implemented a buffer: the hands must maintain the exact same geometric letter for roughly 300 milliseconds. Once validated, the letter is added to the word, and a cool-down block triggers so it doesn't double-type immediately.

### 6. `src/core/motion_predictor.py` (Advanced DL - LSTM Prediction)
**What it does:** Uses a PyTorch Long Short-Term Memory (LSTM) recurrent neural network to predict the *future* body pose based on historical movement. It takes the last 20 frames of 9 critical joint angles and predicts the trajectory for the next 5 frames into the future in real-time.
### Phase 4: Future Motion Forecasting (Deep Learning)
- **Model:** PyTorch LSTM Recurrent Neural Network.
- **Features:** Processes 20-frame sequence of 9 joint angles.
- **Inference:** Forecasts next 5 frames ($165ms$) of motion in real-time.
- **Validation:** Live accuracy scoring compares past forecasts to current actual pose.

### Phase 5: Perfection Pass (Premium HCI)
- **Neon HUD:** Custom glow-effect skeleton drawing replacing standard dots.
- **Perfect Form Heuristics:** 
  - **Squat Depth:** Geometric validation ensuring hips break the knee plane.
  - **Posture Monitor:** Spine alignment slope analysis detecting slouching.
- **Stability Engine:** 5-frame temporal consensus buffer for jitter-free gestures.
- **Real-time Feedback:** Dashboard "Toast" notifications for exercise form alerts.
**Design Decision - Synthetic Pre-training vs Massive Datasets:** To prove the advanced DL concept without requiring gigabytes of Mocap bounding boxes, the `MotionLSTM` prototype is pre-trained on a procedurally generated synthetic dataset of human-like sinusoidal joint kinematics via `train_motion_model.py`. This proves the sequence-to-sequence architectural capability, seamlessly mapping `(batch, 20, 9)` historical tensors to `(batch, 5, 9)` future predictions alongside the live MediaPipe stream at 30 FPS.

### 7. `app.py`, `src/web/stream.py`, and Web Frontend (Flask Architecture)
**What it does:** The API and frontend interface.
**Design Decision - Multi-Threading & MJPEG Streaming:** Standard Flask blocks when serving long-running requests. To make this work, the camera capture and the Heavy CV Engine processing are spun off into a `daemon` background thread (managed explicitly via `src/web/stream.py` `StreamManager`).
*   **The Video Feed:** Handled via HTTP `multipart/x-mixed-replace`. This means the server continuously pushes just JPEGs to an `<img>` tag in the DOM, creating a live video stream without websockets.
*   **The Data APIs:** Since the computer vision runs continuously, the frontend JavaScript (`camera.js`, `game.js`, etc.) repeatedly hits standard REST JSON endpoints (like `/api/analytics` or `/api/game_state`) 4-10 times a second to pull the latest angles/gestures/coordinates/predictions generated by the background thread, effortlessly decoupling the UI rendering layer from the intensive CV calculations.

### 8. The Gesture Game (`templates/game.html` & `static/js/game.js`)
**What it does:** An HTML5 Canvas game where you dodge incoming obstacles by physically moving your body.
**Design Decision - Translating Body to Screen:** I mapped specific physical actions directly to the game state via JavaScript polling:
*   Physical Lean Left/Right = Avatar moves Left/Right.
*   Raised Arms (both wrists higher than shoulders) = Avatar Jumps over low boxes.
*   Crouch down (knees below hip level threshold) = Avatar Ducks under high obstacles.
This creates completely controller-less HCI (Human-Computer Interaction).

---

## Modifying the Code (Examples)

**How to add a new physical exercise (e.g., a "Lateral Raise"):**
1. Open `src/core/exercise.py`.
2. Notice the logic maps the `shoulder` angle. You'd add a new dictionary entry under `self.exercises['lateral_raise']` tracking the `right_shoulder` and `left_shoulder` angles from `pose_data`.
3. Set your angle thresholds (e.g., `thresh_up`: 80°, `thresh_down`: 20°). The state machine automatically handles the rest!

**How to add a new static ASL letter (e.g., the letter 'U'):**
1. Open `src/core/sign_language.py`.
2. Look at the finger boolean states. 'U' is structurally: Index and Middle extended, tightly together, with the thumb pinning the folded ring and pinky fingers.
3. Write the conditional: `if not thumb and index and middle and not ring and not pinky: return 'U', 0.85`.

---

This covers the entirety of the system logic, why we chose specific architectures (rule-based over ML, Threaded Flask processing over single processes, EMA smoothing over raw landmarks), and how a developer can dive right in to extending the code!
