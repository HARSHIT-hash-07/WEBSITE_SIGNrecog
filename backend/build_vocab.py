#!/usr/bin/env python3
"""
build_vocab.py — One-off script to extract the PHOENIX14T gloss vocabulary
from the training data pickle files and write src_vocab.txt.

Run from backend/ directory:
    python3 build_vocab.py
"""
import sys
import os
import pickle
import re
from collections import Counter

# Paths relative to this script (backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "sign_idd_model_20260121_171210")

# Pickle files that contain {'name':..., 'gloss':..., 'sign':...} dicts
SKEL_FILES = [
    os.path.join(MODEL_DIR, "phoenix14t.skels.dev"),
    os.path.join(MODEL_DIR, "phoenix14t.skels.test"),
]

OUTPUT_VOCAB = os.path.join(os.path.dirname(__file__), "src_vocab.txt")

def load_gloss_tokens(path):
    tokens = []
    try:
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        for entry in data:
            gloss = entry.get("gloss", "")
            if gloss and gloss != "unknown":
                # Glosses are space-separated strings like "ICH BIN HEUTE HIER"
                words = gloss.strip().split()
                tokens.extend(words)
        print(f"  Loaded {len(data)} entries from {os.path.basename(path)}")
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
    return tokens

def main():
    print("Building PHOENIX14T gloss vocabulary...")
    
    all_tokens = []
    for skel_file in SKEL_FILES:
        if os.path.exists(skel_file):
            toks = load_gloss_tokens(skel_file)
            all_tokens.extend(toks)
        else:
            print(f"  Skipping (not found): {skel_file}")
    
    if not all_tokens:
        print("ERROR: No tokens found. Check that .skels.dev/.skels.test exist.")
        sys.exit(1)
    
    # Count and sort by frequency (most frequent first)
    counter = Counter(all_tokens)
    sorted_tokens = sorted(counter.keys(), key=lambda t: (-counter[t], t))
    
    print(f"\nTotal unique glosses: {len(sorted_tokens)}")
    print(f"Top 20 most common: {sorted_tokens[:20]}")
    
    # Special tokens come first (matching vocabulary.py order)
    specials = ["<unk>", "<pad>", "<s>", "</s>"]
    # Filter out any specials if they appeared in data
    data_tokens = [t for t in sorted_tokens if t not in specials]
    
    final_vocab = specials + data_tokens
    
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        for token in final_vocab:
            f.write(token + "\n")
    
    print(f"\nVocabulary written to: {OUTPUT_VOCAB}")
    print(f"Total vocab size: {len(final_vocab)} tokens (4 special + {len(data_tokens)} gloss)")

if __name__ == "__main__":
    main()
