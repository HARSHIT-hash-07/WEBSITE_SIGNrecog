import sys
import os
import torch
import numpy as np

# Add the model directory to sys.path so we can import from it
# Assuming directory structure:
# WEBSITE/
#   backend/
#     model_loader.py
#   sign_idd_model_20260121_171210/
#     Sign-IDD/
#       text_to_sign.py

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_ROOT = os.path.join(PROJECT_ROOT, "sign_idd_model_20260121_171210", "Sign-IDD")

if MODEL_ROOT not in sys.path:
    sys.path.append(MODEL_ROOT)

try:
    from text_to_sign import TextToSignPipeline
except ImportError as e:
    print(f"Error importing TextToSignPipeline: {e}")
    # Fallback or error handling
    TextToSignPipeline = None

class SignModel:
    def __init__(self):
        print("Initializing SignBridge Model (Real Mode)...")
        
        if TextToSignPipeline is None:
            raise RuntimeError("Could not load TextToSignPipeline. Check paths.")

        self.config_path = os.path.join(MODEL_ROOT, "Configs", "TA_model.yaml")
        self.ckpt_path = os.path.join(MODEL_ROOT, "Models", "TA_checkpoint", "best.ckpt")
        
        if not os.path.exists(self.config_path):
             print(f"Warning: Config not found at {self.config_path}")
        if not os.path.exists(self.ckpt_path):
             print(f"Warning: Checkpoint not found at {self.ckpt_path}")

        # Initialize the pipeline
        # The Sign-IDD config uses relative paths (e.g., ./Configs/src_vocab.txt)
        # so we must change CWD to MODEL_ROOT during initialization
        
        original_cwd = os.getcwd()
        try:
            os.chdir(MODEL_ROOT)
            
            self.pipeline = TextToSignPipeline(
                config_path=self.config_path,
                checkpoint_path=self.ckpt_path
            )
            self.is_loaded = True
            print("SignBridge Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.is_loaded = False
            raise e
        finally:
            os.chdir(original_cwd)

    def inference(self, text: str):
        """
        Run inference using the pipeline.
        Returns a list of frames, where each frame is a list of joints [x, y, z].
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        print(f"Generating sign for: {text}")
        
        # Run pipeline
        # result['skeleton'] is a numpy array (frames, 150)
        # We need to reshape it to (frames, 50, 3) or similar for the frontend
        # The frontend expects [frames][joints][xyz] -> number[][][]
        
        try:
            # We explicitly pass text. The pipeline handles translation to gloss if needed.
            # But the pipeline assumes German text input for 'text' arg.
            # If we want to support English, we might need to rely on its internal keyword extraction 
            # or add a translation step. The TextToGloss class in text_to_sign.py has some English mapping.
            
            # Generate a unique video name using timestamp or UUID if needed, but for now simple counter or text hash
            # pipeline.run uses `video_name` argument. default is "sign_output".
            video_name = f"sign_{hash(text) % 10000}"
            
            output_dir = os.path.join(CURRENT_DIR, "output")
            result = self.pipeline.run(
                text=text, 
                output_dir=output_dir,
                video_name=video_name
            )
            skeleton = result.get('skeleton')
            video_full_path = result.get('video_path')
            
            if skeleton is None:
                return {"skeletons": [], "video_url": None}

            # Skeleton shape is (frames, 150)
            # 150 = 50 joints * 3 coordinates
            frames = skeleton.shape[0]
            reshaped_skeleton = skeleton.reshape(frames, 50, 3)
            
            return {
                "skeletons": reshaped_skeleton.tolist(),
                "video_url": f"/static/{video_name}.mp4" 
            }

        except Exception as e:
            print(f"Inference error: {e}")
            raise e

