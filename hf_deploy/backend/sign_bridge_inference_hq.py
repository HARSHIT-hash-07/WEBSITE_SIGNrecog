import os
import sys
import torch
import numpy as np
from typing import List, Dict

# Resolve imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from .sign_bridge_inference import SignBridgeInference
    from .constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
except (ImportError, ValueError):
    from sign_bridge_inference import SignBridgeInference
    from constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

class SignBridgeInferenceHQ(SignBridgeInference):
    """
    High Fidelity version of the SignBridge Inference Engine.
    Uses original uncompressed weights and a 'HQ Sampler' tuned for motion.
    """
    
    def translate(self, text: str, sampling_steps: int = 90) -> List[List[List[float]]]:
        """
        Translates text to HQ skeletons using the high-fidelity sampler.
        """
        glosses = self.text_to_glosses(text)
        if not glosses:
             return []

        # Map glosses to indices
        tokens = [BOS_TOKEN] + glosses + [EOS_TOKEN]
        indices = [self.vocab.stoi[t] for t in tokens]
        
        dev = self.device
        src_tensor = torch.tensor([indices], dtype=torch.long, device=dev)
        src_mask = (src_tensor != self.vocab.stoi[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        src_lengths = torch.tensor([len(indices)], dtype=torch.long, device=dev)

        # 1. Encode source
        with torch.no_grad():
            encoder_output = self.model.encode(src_tensor, src_lengths, src_mask)

        # 2. HQ Dynamic Frame Estimation
        # We increase the frames per word to allow for more fluid motion
        # Validation videos usually have approx 100-200 frames for a sentence
        n_frames = max(80, len(glosses) * 20 + 30) 
        
        trg_mask = torch.ones((1, 1, n_frames), device=dev, dtype=torch.bool)
        
        # 3. HQ Sampling with deterministic/stochastic blend
        # Note: We can manually call ddim_sample or use our own loop for better variance control
        # Currently, we'll use the model's ddim_sample but with HQ steps
        print(f"HQ Sampler: Generating {n_frames} frames over {sampling_steps} steps...")
        
        with torch.no_grad():
            # Create a mock input_3d just for shape
            mock_input_3d = torch.zeros((1, n_frames, 150), device=dev)
            
            # The LARGE model typically performs better at 80-100 steps
            raw_skels = self.model.ACD.ddim_sample(
                encoder_output, 
                mock_input_3d, 
                src_mask, 
                trg_mask, 
                sampling_steps=sampling_steps
            )
            
            # Use the final prediction (x0)
            raw_skel = raw_skels[-1][0] # (T, 150)
            
            # HQ MOTION CALIBRATION:
            # If the model is slightly shy, we can apply a very subtle Dynamic Range expansion
            # skel_std = raw_skel.std()
            # if skel_std < 0.15:
            #     raw_skel = (raw_skel - raw_skel.mean()) * 1.2 + raw_skel.mean()

            return raw_skel.reshape(n_frames, 50, 3).tolist()

    def text_to_glosses(self, text: str) -> List[str]:
        # Reuse base class preprocessing
        return super().text_to_glosses(text)
