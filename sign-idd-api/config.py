# sign-idd-api/config.py

import os

# ── User configures this one line ────────────────────────────────────
# Path to the sign_idd_model folder downloaded from Google Drive
# Can also be set via environment variable: MODEL_DIR
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "../sign_idd_model_20260121_171210"  # default: folder sits next to sign-idd-api/
)

# ── Derived paths (no need to change these) ──────────────────────────
VIDEOS_DIR      = os.path.join(MODEL_DIR, "videos")
TEST_VIDEOS_DIR = os.path.join(MODEL_DIR, "test_videos")
DEV_SKELS_PATH  = os.path.join(MODEL_DIR, "phoenix14t.skels.dev")
TEST_SKELS_PATH = os.path.join(MODEL_DIR, "phoenix14t.skels.test")
LOOKUP_PATH     = os.path.join(MODEL_DIR, "lookup.json")