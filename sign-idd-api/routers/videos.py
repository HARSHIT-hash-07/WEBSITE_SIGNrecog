from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from schemas.response import VideoListResponse, SearchResponse
from utils.index import get_all_videos, search_videos, get_video_path

router = APIRouter()

@router.get("/videos", response_model=VideoListResponse)
def list_videos():
    """List all available pre-generated sign language videos."""
    videos = get_all_videos()
    return VideoListResponse(total=len(videos), videos=videos)

@router.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., description="Search term to match against video names")):
    """Fuzzy search videos by name. E.g. /search?q=tagesschau"""
    results = search_videos(q)
    return SearchResponse(query=q, total_matches=len(results), results=results)

@router.get("/video/{name}")
def get_video(name: str):
    """Fetch a specific video by its exact name."""
    path = get_video_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Video '{name}' not found. Use /videos to list all available names.")
    return FileResponse(path, media_type="video/mp4", filename=f"{name}.mp4")