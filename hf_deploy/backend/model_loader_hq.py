import os
import sys
import threading
from typing import Dict, Any, List

# Add backend directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from .sign_bridge_inference_hq import SignBridgeInferenceHQ
except (ImportError, ValueError):
    from sign_bridge_inference_hq import SignBridgeInferenceHQ

# HQ Weights are stored in a dedicated directory
MODEL_ROOT_HQ = os.path.join(os.path.dirname(CURRENT_DIR), "weights_hq")
CONFIG_PATH_HQ = os.path.join(os.path.dirname(CURRENT_DIR), "model_configs", "Sign-IDD-HQ.yaml")

class SignModelHQ:
    """
    Singleton wrapper for the High Fidelity SignBridgeInference engine.
    Uses the uncompressed 1.1GB weights and optimized motion sampling.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SignModelHQ, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        print("Initializing SignBridge HQ Model (High Fidelity Path)...")
        self.engine = None
        self.is_loaded = False
        self._load_error = None
        self._initialized = True
        
        # Load in background
        threading.Thread(target=self._load_model_async, daemon=True).start()

    def _load_model_async(self):
        try:
            # 1. Ensure weight directory exists
            os.makedirs(MODEL_ROOT_HQ, exist_ok=True)
            weight_path = os.path.join(MODEL_ROOT_HQ, "best.ckpt")

            # 2. Check if weights need to be downloaded (Runtime bypass for 1GB repo limit)
            if not os.path.exists(weight_path):
                print(f"HQ Weights not found at {weight_path}. Attempting download from Hub...")
                from huggingface_hub import hf_hub_download
                
                repo_id = os.environ.get("HF_REPO_ID_HQ", "Harshit2907/SignBridge-Weights")
                token = os.environ.get("HF_TOKEN") # Optional: needed if repo is private
                
                print(f"Downloading HQ Weights from {repo_id}...")
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="best.ckpt",
                    local_dir=MODEL_ROOT_HQ,
                    token=token
                )
                print(f"✅ Download complete: {downloaded_file}")

            # 3. Initialize the HQ-specific inference engine
            self.engine = SignBridgeInferenceHQ(MODEL_ROOT_HQ)
            self.is_loaded = True
            print("✅ SignBridge HQ Model loaded and ready for high-fidelity inference.")
        except Exception as e:
            self._load_error = str(e)
            print(f"❌ Failed to load SignBridge HQ Model: {e}")
            import traceback
            traceback.print_exc()

    def inference(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            if self._load_error:
                raise RuntimeError(f"HQ Model failed to load: {self._load_error}")
            raise RuntimeError("HQ Model is still loading. Please try again in 30 seconds.")

        print(f"HQ Inference Request: '{text}'")
        
        try:
            # HIGH FIDELITY PARAMS:
            # We use 90 steps (matching original training) and potentially different guidance or length heuristics
            skeletons = self.engine.translate(text, sampling_steps=90)
            
            import uuid
            from video_renderer import render_skeleton_to_video
            
            filename = f"hq_gen_{uuid.uuid4().hex[:8]}.mp4"
            output_dir = os.path.join(CURRENT_DIR, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # Use standard renderer (stable)
            render_skeleton_to_video(skeletons, output_path, mode="hq")
            
            # URL resolution (assumes same static mount)
            video_url = f"https://harshit2907-sign-idd-inference.hf.space/static/{filename}"
            if os.environ.get("LOCAL_DEV"):
                 video_url = f"http://127.0.0.1:8001/static/{filename}"

            return {
                "skeletons": None,
                "video_url": video_url,
                "glosses": self.engine.text_to_glosses(text)
            }
        except Exception as e:
            print(f"HQ Inference error: {e}")
            raise e

# Global singleton instance for HQ
sign_model_hq = SignModelHQ()
