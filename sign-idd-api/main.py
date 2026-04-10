from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import videos
from utils.index import load_index
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    yield

app = FastAPI(
    title="Sign-IDD API",
    description="API to serve pre-generated sign language videos from the Sign-IDD model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(videos.router)

@app.get("/health")
def health():
    return {"status": "ok"}