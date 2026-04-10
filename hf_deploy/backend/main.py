import sys
import os

# Add backend directory to sys.path to resolve imports in both Docker and Local runs
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

try:
    from .model_loader import SignModel
    from .model_loader_hq import sign_model_hq
except (ImportError, ValueError):
    from model_loader import SignModel
    from model_loader_hq import sign_model_hq

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="SignBridge API", version="1.0.0")

# CORS Configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Initialize Model
model = SignModel()

class TranslationRequest(BaseModel):
    text: str
    gloss_mode: Optional[str] = "default"

class TranslationResponse(BaseModel):
    skeletons: Optional[List[List[List[float]]]] = None # [frames][joints][xyz]
    video_url: Optional[str] = None
    text_processed: str

@app.get("/")
async def root():
    return {"message": "SignBridge AI API is running"}

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        result = model.inference(request.text)
        return {
            "skeletons": result.get("skeletons", []),
            "video_url": result.get("video_url"),
            "text_processed": request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/translate_hq", response_model=TranslationResponse)
async def translate_text_hq(request: TranslationRequest):
    try:
        result = sign_model_hq.inference(request.text)
        return {
            "skeletons": result.get("skeletons", []),
            "video_url": result.get("video_url"),
            "text_processed": request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
