import time
import sys
import traceback

try:
    from src.web.stream import StreamManager
    
    print("Testing StreamManager...")
    sm = StreamManager()
    sm.start()
    
    for i in range(10):
        frame = sm.get_frame_jpeg()
        if frame is not None:
            print(f"Got frame! len={len(frame)}")
            break
        time.sleep(1)
        print(f"Wait {i+1}...")
        
    if sm._thread and not sm._thread.is_alive():
        print("Thread died! Something crashed in the background thread.")
        
    sm.stop()
    print("Test complete.")
except Exception as e:
    print("Exception during test:")
    traceback.print_exc()
