import time
import sys
import traceback
import numpy as np
import cv2

try:
    from src.web.stream import StreamManager
    
    print("Testing StreamManager with mocked frame...")
    sm = StreamManager()
    
    # Mock the capture loop directly to avoid VideoCapture conflicts
    success = True
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    sm._running = True
    
    # Run a few iterations of the loop manually to trace exceptions
    for i in range(25):
        # 1. Pose
        pose_data = {'landmarks': {}, 'angles': {}, 'velocities': {},
                     'bbox': None, 'confidence': 0.0}
        
        # Mock some angles to populate the feature vec
        angles = {
            'left_elbow': 45.0, 'right_elbow': 45.0,
            'left_shoulder': 90.0, 'right_shoulder': 90.0,
            'left_hip': 180.0, 'right_hip': 180.0,
            'left_knee': 180.0, 'right_knee': 180.0,
            'neck_inclination': 0.0
        }
        
        ordered_keys = [
            'left_elbow', 'right_elbow',
            'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'neck_inclination'
        ]
        feature_vec = [angles.get(k, 0.0) or 0.0 for k in ordered_keys]
        sm._history.append(feature_vec)
        prediction = sm.motion_predictor.predict(list(sm._history))
        pose_data['motion_prediction'] = prediction
        
        print(f"Frame {i+1}: Prediction type={type(prediction)}")
        if prediction is not None:
            print(f"Prediction length: {len(prediction)}")
            
    print("Test complete without crashes.")
except Exception as e:
    print("Exception during test:")
    traceback.print_exc()
