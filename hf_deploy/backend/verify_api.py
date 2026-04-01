import sys
import os
# Add current directory to path
sys.path.append(os.getcwd())

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Testing API...")
try: 
    response = client.post("/translate", json={"text": "hallo"})
    if response.status_code == 200:
        data = response.json()
        print("Success! Status 200")
        if "skeletons" in data and isinstance(data["skeletons"], list):
            print(f"Response contains {len(data['skeletons'])} frames")
            
            if "video_url" in data and data["video_url"]:
                print(f"Response contains video URL: {data['video_url']}")
            else:
                print("Warning: Response missing video_url")

            # check shape of first frame
            if len(data['skeletons']) > 0:
                 frame = data['skeletons'][0]
                 print(f"Frame 0 has {len(frame)} joints")
                 # expected ~50 joints * 3 coords based on my model loader logic, or whatever model returns
                 # The model loader reshapes to (frames, 50, 3)
                 if len(frame) == 50 and len(frame[0]) == 3:
                     print("Skeleton shape is correct: (frames, 50, 3)")
                 else:
                     print(f"Warning: Unexpected joint count/dims: {len(frame)}")
        else:
            print("Error: Missing 'skeletons' key or invalid format")
    else:
        print(f"Failed with status {response.status_code}: {response.text}")

except Exception as e:
    print(f"Exception during test: {e}")
