import urllib.request
import json

try:
    print("Testing /api/analytics...")
    req = urllib.request.urlopen("http://localhost:5000/api/analytics", timeout=5)
    data = json.loads(req.read().decode())
    print("Response keys:", list(data.keys()))
    if data:
        print("Pose keys:", list(data.get('pose', {}).keys()))
        print("Has prediction:", 'motion_prediction' in data.get('pose', {}))
    else:
        print("Empty response!")
except Exception as e:
    print("Error fetching analytics:", e)
