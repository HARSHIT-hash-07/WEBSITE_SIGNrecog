# coding: utf-8
"""
Shared constants for Sign-IDD model code.
These are the standard special tokens used by the PHOENIX14T vocabulary.
"""

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# Pad value for target skeleton sequences (used in loss masking)
TARGET_PAD = 0.0

# Default index for unknown tokens (callable so defaultdict works)
DEFAULT_UNK_ID = lambda: 0
