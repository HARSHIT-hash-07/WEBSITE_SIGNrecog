from pydantic import BaseModel
from typing import List, Optional

class VideoEntry(BaseModel):
    name: str
    video_url: str

class VideoListResponse(BaseModel):
    total: int
    videos: List[VideoEntry]

class SearchResponse(BaseModel):
    query: str
    total_matches: int
    results: List[VideoEntry]

class ErrorResponse(BaseModel):
    detail: str