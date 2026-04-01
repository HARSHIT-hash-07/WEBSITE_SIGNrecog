# coding: utf-8
import torch
import torch.nn.functional as F
from typing import Any, Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import TARGET_PAD


class Batch:
    """
    Batch wrapper that normalizes different batch formats.

    Accepts either:
      - legacy torchtext batch object with attributes:
            .src (tensor), .src_lengths (tensor or int), .trg (tensor), .file_paths (list)
      - modern tuple from DataLoader/collate_fn:
            (src_padded, src_lengths, trg_padded, files)

    Args:
        torch_batch: batch object or tuple from DataLoader
        pad_index: integer index used for source padding
        model: model instance (used to check use_cuda)
    """
    def __init__(self, torch_batch: Any, pad_index: int, model: Any):
        # Initialize common attributes
        self.src = None             # LongTensor [B, S]
        self.src_lengths = None     # LongTensor [B]
        self.src_mask = None        # Bool/ByteTensor [B,1,S]
        self.nseqs = 0

        self.trg = None             # FloatTensor [B, T, trg_size]
        self.trg_input = None       # FloatTensor [B, T, trg_size] (for teacher forcing / model input)
        self.trg_mask = None        # Bool/ByteTensor [B,1,T] (True where not padded)
        self.trg_lengths = None     # int or LongTensor
        self.ntokens = 0            # number of non-pad target frames (sum)
        self.file_paths: List[str] = []

        # model flags
        self.use_cuda = getattr(model, "use_cuda", False)
        self.target_pad = TARGET_PAD

        # Unpack depending on batch format
        # Handle legacy torchtext-like object with attributes
        if hasattr(torch_batch, "src") and hasattr(torch_batch, "file_paths"):
            # torchtext-style Example batch
            try:
                # src may be (src_tensor, src_lengths) or src_tensor alone
                if isinstance(torch_batch.src, tuple) or isinstance(torch_batch.src, list):
                    self.src, self.src_lengths = torch_batch.src
                else:
                    self.src = torch_batch.src
                    # try to obtain lengths if available on batch
                    self.src_lengths = getattr(torch_batch, "src_lengths", torch.tensor([s.size(0) for s in self.src], dtype=torch.long))
            except Exception:
                # fallback: assume src is tensor and compute lengths from padding
                self.src = torch_batch.src
                self.src_lengths = getattr(torch_batch, "src_lengths", torch.sum(self.src != pad_index, dim=1))

            self.file_paths = list(getattr(torch_batch, "file_paths", []))

            # Targets (if present)
            if hasattr(torch_batch, "trg"):
                self.trg = torch_batch.trg
        # Handle tuple produced by DataLoader / collate_fn
        elif isinstance(torch_batch, (tuple, list)) and len(torch_batch) >= 3:
            # Expected format: (src_padded, src_lengths, trg_padded, files)
            # Some collate_fns may not return src_lengths; handle both cases.
            # Common expected:
            #   src_padded: LongTensor [B, S]
            #   src_lengths: LongTensor [B]
            #   trg_padded: FloatTensor [B, T, trg_size]
            #   files: list[str]
            try:
                self.src = torch_batch[0]
                self.src_lengths = torch_batch[1]
                self.trg = torch_batch[2]
                # files may be absent or None
                if len(torch_batch) > 3:
                    self.file_paths = list(torch_batch[3])
                else:
                    self.file_paths = []
            except Exception:
                raise ValueError("Unrecognized tuple batch format. Expected (src, src_lengths, trg, files).")
        else:
            raise ValueError("Unrecognized batch format passed to Batch.")

        # Ensure shapes / dtypes
        if isinstance(self.src_lengths, int):
            self.src_lengths = torch.tensor([self.src_lengths] * self.src.size(0), dtype=torch.long)
        if self.src is not None and not isinstance(self.src_lengths, torch.Tensor):
            # attempt to compute lengths from padding if possible
            try:
                self.src_lengths = torch.sum(self.src != pad_index, dim=1).to(torch.long)
            except Exception:
                # fallback zeros
                self.src_lengths = torch.zeros(self.src.size(0), dtype=torch.long)

        # src_mask: True where token != pad_index
        if self.src is not None:
            self.src_mask = (self.src != pad_index).unsqueeze(1)  # [B,1,S]
            self.nseqs = self.src.size(0)

        # Targets handling
        if self.trg is not None:
            # trg is expected shape [B, T, trg_size]
            # trg_lengths: number of frames (T) - if not available infer from shape
            try:
                self.trg_lengths = self.trg.shape[1]
            except Exception:
                self.trg_lengths = None

            # trg_input: the model expects target input. Keep same shape as trg.
            # If you want to shift / remove last frame, do that in the training loop or here by uncommenting the next line:
            # self.trg_input = self.trg[:, :-1, :].clone()
            self.trg_input = self.trg.clone()

            # trg mask: True where frame is not padding. We assume padding frames have all elements equal to TARGET_PAD.
            # To detect padded frames, compare first element of each frame to TARGET_PAD (fast and typical).
            # Fallback: compare sum across frame (if consistent).
            try:
                # If last dim exists
                if self.trg.dim() == 3:
                    self.trg_mask = (self.trg[:, :, 0] != self.target_pad).unsqueeze(1)  # [B,1,T]
                else:
                    # unexpected dimensions -> assume all valid
                    self.trg_mask = torch.ones((self.trg.size(0), 1, self.trg.size(1)), dtype=torch.bool)
            except Exception:
                # fallback: assume all frames valid
                self.trg_mask = torch.ones((self.trg.size(0), 1, self.trg.size(1)), dtype=torch.bool)

            # ntokens: number of non-padding frames across the batch
            try:
                self.ntokens = int(torch.sum(self.trg_mask).item())
            except Exception:
                self.ntokens = 0

        # Filepaths: ensure list
        if self.file_paths is None:
            self.file_paths = []

        # Move to GPU if required
        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """Move the batch tensors to GPU (in-place)."""
        if self.src is not None:
            self.src = self.src.to(device)
            self.src_mask = self.src_mask.to(device)
            self.src_lengths = self.src_lengths.to(device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device)
        if self.trg is not None:
            self.trg = self.trg.to(device)
        if self.trg_mask is not None:
            self.trg_mask = self.trg_mask.to(device)

    def to(self, device_: torch.device):
        """Move batch to specified device and return self (convenience)."""
        if self.src is not None:
            self.src = self.src.to(device_)
            self.src_mask = self.src_mask.to(device_)
            self.src_lengths = self.src_lengths.to(device_)
        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device_)
        if self.trg is not None:
            self.trg = self.trg.to(device_)
        if self.trg_mask is not None:
            self.trg_mask = self.trg_mask.to(device_)
        return self
