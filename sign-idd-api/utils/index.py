import os
from typing import List, Dict

VIDEO_DIR = "test_videos"

def build_video_index() -> Dict[str, str]:
    """
    Scans test_videos/ and builds a dict: { name -> filepath }
    e.g. { "25October_2010_Monday_tagesschau-17": "test_videos/25October_2010_Monday_tagesschau-17.mp4" }
    """
    index = {}
    if not os.path.exists(VIDEO_DIR):
        return index
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(".mp4"):
            name = filename.replace(".mp4", "")
            index[name] = os.path.join(VIDEO_DIR, filename)
    return index

# Loaded once at startup
VIDEO_INDEX: Dict[str, str] = {}

def load_index():
    global VIDEO_INDEX
    VIDEO_INDEX = build_video_index()
    print(f"[Sign-IDD API] Indexed {len(VIDEO_INDEX)} videos from {VIDEO_DIR}/")

def get_all_videos() -> List[dict]:
    return [{"name": name, "video_url": f"/video/{name}"} for name in sorted(VIDEO_INDEX.keys())]

def search_videos(query: str) -> List[dict]:
    q = query.lower()
    matches = [
        {"name": name, "video_url": f"/video/{name}"}
        for name in sorted(VIDEO_INDEX.keys())
        if q in name.lower()
    ]
    return matches

def get_video_path(name: str) -> str | None:
    return VIDEO_INDEX.get(name, None)