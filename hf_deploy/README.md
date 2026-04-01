---
title: SignBridge AI
emoji: 🤟
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# SignBridge AI: Live AI Sign Language Generation

This is the backend API for SignBridge, providing real-time text-to-sign skeletal animation generation.

## How to use

The API is built with FastAPI. You can interact with it via POST requests to `/translate`.

### Endpoint: `/translate`

**POST Request:**
```json
{
  "text": "today weather rain"
}
```

**Response:**
```json
{
  "skeletons": null,
  "video_url": "http://127.0.0.1:8000/static/gen_abc123.mp4",
  "text_processed": "today weather rain"
}
```

## Deployment Notes

- This space requires `best.ckpt` to be placed in the `weights/` directory before building.
- The model runs on CPU by default for stability in free-tier spaces.
- Uses FFmpeg for video processing.
