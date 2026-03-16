import time
import sys
import traceback
import cv2

try:
    from src.web.stream import StreamManager
    
    print("Testing StreamManager with actual Webcam...")
    sm = StreamManager()
    sm.start()
    
    # Wait for the thread to initialize
    time.sleep(2)
    
    if sm._thread and not sm._thread.is_alive():
        print("Thread died during startup.")
    else:
        print("Thread is alive. Trying to grab a frame...")
        
    for i in range(10):
        frame = sm.get_frame_jpeg()
        if frame is not None:
            print(f"SUCCESS! Got frame {i+1}, length={len(frame)}")
            break
        print(f"Wait {i+1}... no frame yet")
        time.sleep(1)
        
    print("Checking analytics:")
    print(sm.get_analytics())
    
    sm.stop()
    print("Test complete.")
except Exception as e:
    print("Exception during test:")
    traceback.print_exc()
