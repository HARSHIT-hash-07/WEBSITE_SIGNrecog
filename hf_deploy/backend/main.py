from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model_loader import SignModel

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
