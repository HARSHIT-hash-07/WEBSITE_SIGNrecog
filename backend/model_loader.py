import os
import sys
import threading
from typing import Dict, Any, List

# Add backend to path so we can import sign_bridge_inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from sign_bridge_inference import SignBridgeInference

# Configuration
MODEL_ROOT = "/Users/harshit/Documents/WEBSITE_EXPLO/sign_idd_model_20260121_171210"

class SignModel:
    """
    Singleton wrapper for the SignBridgeInference engine.
    Handles thread-safe inference and model lifecycle.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SignModel, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        print("Initializing SignBridge Model (Real Integration)...")
        self.engine = None
        self.is_loaded = False
        self._load_error = None
        self._initialized = True
        
        # Load the model in a background thread to avoid blocking FastAPI startup
        threading.Thread(target=self._load_model_async, daemon=True).start()

    def _load_model_async(self):
        """
        Loads the model weights into memory.
        """
        try:
            # We use the inference engine we just built
            self.engine = SignBridgeInference(MODEL_ROOT)
            self.is_loaded = True
            print("✅ SignBridge Model loaded and ready for inference.")
        except Exception as e:
            self._load_error = str(e)
            print(f"❌ Failed to load SignBridge Model: {e}")
            import traceback
            traceback.print_exc()

    def inference(self, text: str) -> Dict[str, Any]:
        """
        Performs inference on the provided text.
        """
        if not self.is_loaded:
            if self._load_error:
                raise RuntimeError(f"Model failed to load: {self._load_error}")
            raise RuntimeError("Model is still loading. Please try again in 30 seconds.")

        print(f"Inference Request: '{text}'")
        
        try:
            # Run the translation!
            # We use 20 sampling steps for a good balance of speed and quality on M2
            skeletons = self.engine.translate(text, sampling_steps=20)
            
            return {
                "skeletons": skeletons,
                "video_url": None, # Diffusion model produces raw skeletons, not video
                "glosses": self.engine.text_to_glosses(text)
            }
        except Exception as e:
            print(f"Inference error: {e}")
            raise e

# Global singleton instance
sign_model = SignModel()
