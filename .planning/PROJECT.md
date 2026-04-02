# SignBridge — Sign Language Translation Platform

## Project Overview
SignBridge is an end-to-end AI-powered sign language translation website. It takes text input (e.g., "today", "weather is cold") and renders an animated 3D skeleton performing the corresponding German Sign Language (DGS) signs using the Sign-IDD diffusion model.

## Architecture
- **Frontend**: Next.js (TypeScript) — text input UI, 3D skeleton renderer, video player
- **Backend**: FastAPI (Python) — `/translate` endpoint serving Sign-IDD inference
- **Model**: Sign-IDD (multi-rate DDIM diffusion, PINN-based) — Ankita's architecture
- **Data**: PHOENIX-14T German weather sign language dataset

## Current Milestone: v1.0 — Motion Variance & Animation Quality

The Sign-IDD model was successfully restored to use the correct multi-rate diffusion architecture (Ankita's PINN code). It produces non-deterministic output, but:
- **Motion variance is too low** (~0.000048 std across time) — animations look nearly static
- **Root cause**: default `sampling_timesteps = 5` in YAML config; inference code ignores the steps parameter passed at runtime
- **Device bug**: `ACD.py` uses module-level `device` variable (not `self.device`) in `ddim_sample()`

## Tech Stack
- Python 3.13, PyTorch, FastAPI
- Next.js 14, React, Three.js
- Sign-IDD model at: `/Users/harshit/Documents/WEBSITE_EXPLO/sign_idd_model_20260121_171210`
- Inference entry: `backend/sign_bridge_inference.py`
- Model architecture: `SignIDD_CodeFiles/ACD.py`, `ACD_Denoiser.py`
- Config: `backend/model_configs/Sign-IDD.yaml`

## Key Constraints
- Model weights are CPU-only (forced; MPS causes runtime crash)
- Model checkpoint has 8 key mismatches (tolerated, non-critical)
- Vocab size: 1,089 tokens (PHOENIX-14T German ASL glosses)
