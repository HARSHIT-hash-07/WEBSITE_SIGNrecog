import os
import sys
import numpy as np
import torch

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from sign_bridge_inference import SignBridgeInference

# POINT TO THE LARGE CHECKPOINT
LARGE_MODEL_PATH = "/Users/harshit/Documents/WEBSITE_EXPLO/sign_idd_model_20260121_171210/best.ckpt"
MODEL_ROOT = os.path.dirname(LARGE_MODEL_PATH)

print(f"Loading LARGE model from: {LARGE_MODEL_PATH}")
# SignBridgeInference expects a weights directory with 'best.ckpt'
engine = SignBridgeInference(MODEL_ROOT)

text = "Today weather rain"
print(f"Translating: {text}")
skeletons = engine.translate(text, sampling_steps=50)

skel_array = np.array(skeletons)
skel_std = np.std(skel_array, axis=0).mean()
skel_mean = np.mean(skel_array)
skel_min = np.min(skel_array)
skel_max = np.max(skel_array)

print("-" * 30)
print(f"Frames: {len(skeletons)}")
print(f"Mean Coordinate Value: {skel_mean:.6f}")
print(f"Min Coord: {skel_min:.6f}, Max Coord: {skel_max:.6f}")
print(f"Average Variance (STD) across frames: {skel_std:.6f}")

if skel_std < 1e-4:
    print("CRITICAL: The LARGE model is also still?!")
else:
    print("SUCCESS: Motion detected in LARGE model!")

from video_renderer import render_skeleton_to_video
output_path = os.path.join(CURRENT_DIR, "debug_large_model.mp4")
render_skeleton_to_video(skeletons, output_path)
print(f"Video rendered to: {output_path}")
