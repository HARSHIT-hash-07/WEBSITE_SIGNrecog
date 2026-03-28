# Architecture

This document maps the architectural patterns, system layers, and data flows of the SignBridge project.

## Overview
SignBridge follows a microservice-based architectural design aimed at cleanly separating computational ML workloads from the client layer. 

## System Layers
1. **Presentation Layer (`/frontend`):**
   - Renders 2D UI and handles 3D rendering of skeletons via Three.js.
   - Manages client-side state using Zustand.
   - Interacts with Supabase for data resilience, routing user interactions asynchronously.
2. **ML Inference Layer (`/backend`):**
   - Manages the heavy lifting. Specifically, the `SignModel` class instantiates the PyTorch model tensors inside `model_loader.py`.
   - On request, processes the translation, generates representations, optionally produces output videos in `/output`, and responds via `main.py`.
3. **Serving Layer (`/sign-idd-api`):**
   - Works as an indexing microservice. Uses `utils/index.py` at application startup to build memory references to pre-processed directories (`sign_idd_model_...`).
   - Reduces latency for static video retrieval without invoking the heavier `SignModel`.

## Data Flow
- **User Request:** Text entered in UI → Next.js standardizes payload.
- **Routing:** 
  - If dynamic translation: Request dispatched to Inference API (`/backend/main.py` -> `translate()`). PyTorch models create data structures and send video URLs and 3D skeleton frames (`[frames][joints][xyz]`) back to client.
  - If static video search: Request dispatched to Sign-IDD API (`/sign-idd-api/...`).
- **Response Handling:** Frontend uses React Three Fiber to visualize the `skeletons` array in real-time, or mounts the `video_url` in a player.
