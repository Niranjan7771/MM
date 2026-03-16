import time
import sys
import traceback
import cv2

try:
    from src.web.stream import StreamManager
    
    print("Testing StreamManager over 30 frames...")
    sm = StreamManager()
    sm.start()
    
    time.sleep(2)
    
    frame_count = 0
    start_time = time.time()
    
    while frame_count < 30 and time.time() - start_time < 15:
        a = sm.get_analytics()
        if a and 'pose' in a and a['pose'].get('motion_prediction') is not None:
            print(f"GOT PREDICTION at checking loop {frame_count}")
            break
        time.sleep(0.1)
        frame_count += 1
        
    if sm._thread and not sm._thread.is_alive():
        print("THREAD CRASHED!")
        
    print("Analytics keys:", list(sm.get_analytics().keys()))
    sm.stop()
    print("Test complete.")
except Exception as e:
    print("Exception during test:")
    traceback.print_exc()
