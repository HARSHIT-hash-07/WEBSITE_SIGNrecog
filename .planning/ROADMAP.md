# Milestone v1.0 — Motion Variance & Animation Quality

## Goal
Fix the Sign-IDD model's low motion variance so the generated skeletal animations are visually diverse, input-dependent, and high-quality. Verify the full pipeline from text input to rendered video on the website.

## Phases

### Phase 1: Fix DDIM Sampling & Device Bugs
**Goal:** Fix the two root-cause bugs blocking animation quality — the ignored sampling_steps parameter and the module-level device bug — and verify measurably higher motion variance.

- Fix `sampling_timesteps` to be configurable at runtime (not locked to YAML default of 5)
- Fix `device` bug in `ACD.py` `ddim_sample()` (lines 197, 204 use module-level `device` instead of tensor device)
- Update `sign_bridge_inference.py` to pass `sampling_steps` correctly
- Update `Sign-IDD.yaml` config to set a sensible default (50+ steps)
- Verify: time variance > 0.001, different inputs produce measurably different outputs

### Phase 2: Full Pipeline End-to-End Verification
**Goal:** Verify the complete path from the FastAPI `/translate` endpoint to the video renderer on the website produces high-quality, diverse animations.

- Start backend and frontend
- Test `/translate` with several different inputs
- Verify rendered videos differ visually and have smooth motion
- Check that skeleton joint positions are within realistic human bounds
