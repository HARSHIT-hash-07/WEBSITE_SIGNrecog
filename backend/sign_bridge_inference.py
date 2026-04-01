import os
import sys
import torch
import numpy as np
import yaml
from typing import List, Dict, Any

# Ensure we can import from SignIDD_CodeFiles
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
CODE_DIR = os.path.join(PROJECT_ROOT, "SignIDD_CodeFiles")

if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from model import build_model
from vocabulary import Vocabulary
from constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
import ACD # Access the module to override its global 'device'
import builders # Access the module to override its global 'device'

class SignBridgeInference:
    def __init__(self, model_root: str, device: str = None):
        self.model_root = model_root
        
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing SignBridgeInference on: {self.device}")
        
        # Override global devices in research modules (Mac-compatibility hack)
        ACD.device = self.device
        builders.device = self.device
        
        # NOTE: Model weights (best.ckpt) are too large for the repo (>1GB).
        # You must provide the path to the folder containing best.ckpt.
        self.ckpt_path = os.path.join(model_root, "best.ckpt")
        
        # Vocab and Config are now tracked in the repository for reproducibility.
        self.vocab_path = os.path.join(BACKEND_DIR, "model_configs", "src_vocab.txt")
        self.config_path = os.path.join(BACKEND_DIR, "model_configs", "Sign-IDD.yaml")
        
        # 1. Load Vocab
        self.vocab = Vocabulary(file=self.vocab_path)
        
        # 2. Load Config
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # 3. Build Model (using the official factory function)
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        
        # trg_vocab is not strictly used for inference mapping, just shared dimensions
        # providing a list of 150 items to match cfg.model.trg_size if necessary
        trg_vocab_dummy = ["dummy"] * 150 
        
        self.model = build_model(
            cfg=self.cfg,
            src_vocab=self.vocab,
            trg_vocab=trg_vocab_dummy,
            checkpoint=checkpoint
        )
        self.model.to(torch.float32)
        self.model.to(self.device).eval()
        print("Model built and weights loaded.")

    def text_to_glosses(self, text: str) -> List[str]:
        text = text.upper().strip()
        for p in ".,!?;:": text = text.replace(p, "")
        words = text.split()
        
        # Expanded mapping from common English descriptors to official PHOENIX-14T German glosses
        mapping = {
            "I": "ICH", 
            "TODAY": "HEUTE", 
            "TOMORROW": "MORGEN", 
            "YESTERDAY": "GESTERN",
            "WEATHER": "WETTER",
            "RAIN": "REGEN", 
            "SUN": "SONNE", 
            "CLOUDS": "WOLKE", 
            "SNOW": "SCHNEE",
            "COLD": "KALT", 
            "WARM": "WARM", 
            "SOUTH": "SUED", 
            "NORTH": "NORD",
            "EAST": "OST", 
            "WEST": "WEST", 
            "NIGHT": "NACHT", 
            "AND": "UND",
            "NICE": "SCHOEN",
            "BAD": "SCHLECHT",
            "STORM": "STURM",
            "TEMPERATURE": "TEMPERATUR",
            "MONDAY": "MONTAG",
            "TUESDAY": "DIENSTAG",
            "WEDNESDAY": "MITTWOCH",
            "THURSDAY": "DONNERSTAG",
            "FRIDAY": "FREITAG",
            "SATURDAY": "SAMSTAG",
            "SUNDAY": "SONNTAG"
        }
        
        res = []
        for w in words:
            mapped = mapping.get(w, w)
            # Check if mapped gloss exists in vocab
            if mapped in self.vocab.stoi:
                res.append(mapped)
            else:
                # If not found, use closest match or <unk>
                # Using <unk> implicitly by mapping to word not in stoi
                res.append(mapped)
        return res

    def translate(self, text: str, sampling_steps: int = 50) -> List[List[List[float]]]:
        glosses = self.text_to_glosses(text)
        tokens = [BOS_TOKEN] + glosses + [EOS_TOKEN]
        indices = [self.vocab.stoi[t] for t in tokens]
        
        src_tensor = torch.LongTensor([indices]).to(self.device)
        src_length = torch.LongTensor([len(indices)]).to(self.device)
        src_mask = (src_tensor != self.vocab.stoi[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        
        # Estimate sequence length: ~15 frames per gloss + 20 buffer
        # PHOENIX videos are typically 60-200 frames.
        n_frames = max(60, len(glosses) * 15 + 20)
        
        with torch.no_grad():
            # a. Encoding
            src_embedded = self.model.src_embed(src_tensor)
            # Match the signature: forward(embed_src, src_length, mask)
            encoder_output = self.model.encoder(src_embedded, src_length, src_mask)
            
            # b. Diffusion Sampling
            # We need dummy input_3d to determine shape (batch, frames, 150)
            dummy_trg = torch.zeros((1, n_frames, 150), device=self.device)
            trg_mask = torch.ones((1, 1, n_frames), device=self.device).bool()
            
            # Use ddim_sample from ACD directly, passing runtime sampling_steps
            results = self.model.ACD.ddim_sample(
                encoder_output=encoder_output,
                input_3d=dummy_trg,
                src_mask=src_mask,
                trg_mask=trg_mask,
                sampling_steps=sampling_steps  # runtime override (uses self.sampling_timesteps if None)
            )
            
            # Take the final prediction
            # Depending on ACD.py logic, it might be the last one in the results list
            skeletons_flat = results[-1] if isinstance(results, list) else results
            
            raw_skel = skeletons_flat.cpu().numpy()[0]
            frames = raw_skel.shape[0]
            return raw_skel.reshape(frames, 50, 3).tolist()
