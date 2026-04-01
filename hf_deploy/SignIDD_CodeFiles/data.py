# coding: utf-8
"""
Modernized data.py
- Removes legacy torchtext.data.Field/Example/BucketIterator usage
- Uses torch.utils.data.Dataset + DataLoader
- Keeps compatibility with original config-driven load_data signature
- Provides make_data_iter that returns a standard PyTorch DataLoader

Replace your existing data.py with this file.
"""
import io
import os
from typing import List, Tuple, Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# If you have a custom build_vocab in vocabulary.py, we'll continue using it
from vocabulary import build_vocab, Vocabulary
from constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD


# ------------------------
# Helper: token -> id
# ------------------------

def _token_to_id(vocab, token):
    """Robust token->index mapping that supports several vocab interfaces.

    Tries common patterns (vocab[token], vocab.stoi[token], dict-like).
    Falls back to 0 if none available.
    """
    try:
        return vocab[token]
    except Exception:
        # try attribute 'stoi'
        try:
            return vocab.stoi[token]
        except Exception:
            # try dict-like get
            try:
                return vocab.get(token)
            except Exception:
                # final fallback: 0
                return 0


# ------------------------
# Dataset
# ------------------------
class SignProdDataset(Dataset):
    """Dataset for sign production. Reads three parallel files: src, trg, files.

    The constructor mirrors the original behaviour but does not rely on torchtext
    Field/Example APIs. It stores lists of raw source strings, target-frame tensors,
    and file paths. Filtering by max_sent_length can be performed by the caller
    (load_data) before returning the dataset.
    """

    def __init__(self, path: str, exts: Tuple[str, str, str], trg_size: int, skip_frames: int = 1):
        self.src: List[str] = []
        self.trg: List[torch.Tensor] = []
        self.files: List[str] = []

        src_path, trg_path, file_path = tuple(os.path.expanduser(path + x) for x in exts)

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
                io.open(file_path, mode='r', encoding='utf-8') as files_file:

            for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
                src_line, trg_line, files_line = src_line.strip(), trg_line.strip(), files_line.strip()

                if not src_line or not trg_line:
                    continue

                # convert target string of floats into frames
                vals = trg_line.split()
                if len(vals) == 0:
                    continue

                try:
                    trg_vals = [float(v) + 1e-8 for v in vals]
                except ValueError:
                    # skip malformed lines
                    continue

                # group into frames of length trg_size, optionally skipping frames
                frames = [trg_vals[i:i + trg_size] for i in range(0, len(trg_vals), trg_size * skip_frames)]

                if len(frames) == 0:
                    continue

                self.src.append(src_line)
                # store targets as float32 tensors: (num_frames, trg_size)
                self.trg.append(torch.tensor(frames, dtype=torch.float32))
                self.files.append(files_line)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.trg[idx], self.files[idx]


# ------------------------
# Collate function
# ------------------------
def collate_fn(batch, vocab, trg_size, lowercase: bool = False, return_lengths: bool = True):
    """Batching + padding.

    - Tokenizes source by whitespace (word-level). Lowercasing optional.
    - Numericalizes using provided vocab object (robust to several interfaces).
    - Pads source sequences with vocab[PAD_TOKEN] and targets with TARGET_PAD.
    - Returns (src_padded, src_lengths, trg_padded, files) if return_lengths True,
      otherwise (src_padded, trg_padded, files).
    """
    src, trg, files = zip(*batch)

    # tokenize
    if lowercase:
        src_tok = [s.lower().split() for s in src]
    else:
        src_tok = [s.split() for s in src]

    # numericalize
    src_ids = []
    for sent in src_tok:
        ids = [_token_to_id(vocab, tok) for tok in sent]
        src_ids.append(torch.tensor(ids, dtype=torch.long))

    # pad sources
    # determine pad index
    try:
        pad_idx = vocab[PAD_TOKEN]
    except Exception:
        try:
            pad_idx = vocab.stoi[PAD_TOKEN]
        except Exception:
            pad_idx = 0

    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_idx) if len(src_ids) > 0 else torch.zeros((0, 0), dtype=torch.long)

    # compute lengths
    src_lengths = torch.tensor([len(s) for s in src_ids], dtype=torch.long)

    # pad targets (already float tensors of shape (num_frames, trg_size))
    trg_padded = pad_sequence(trg, batch_first=True, padding_value=TARGET_PAD) if len(trg) > 0 else torch.zeros((0, 0, trg_size), dtype=torch.float32)

    if return_lengths:
        return src_padded, src_lengths, trg_padded, list(files)
    else:
        return src_padded, trg_padded, list(files)


# ------------------------
# Data loader / iterator builder
# ------------------------
def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False,
                   vocab=None,
                   trg_size: int = 0,
                   lowercase: bool = False) -> Iterator:
    """Return a PyTorch DataLoader that yields batches as tuples.

    The returned iterator yields tuples produced by collate_fn:
      (src_padded, src_lengths, trg_padded, files)

    Notes:
      - If `vocab` is not provided, the function will attempt to use dataset.vocab.
      - Token-aware ('token') batching is not implemented here and falls back to sentence batching.
    """
    # normalize batch_type
    if batch_type == "token":
        # token-aware dynamic batching not implemented; fallback to sentence
        batch_type = "sentence"

    # determine shuffle behaviour: when training, default to shuffle unless explicitly False
    actual_shuffle = True if train else shuffle

    # allow vocab to be provided either as argument or attached to the dataset
    if vocab is None and hasattr(dataset, "vocab"):
        vocab = getattr(dataset, "vocab")

    # collate requires vocab and trg_size (trg_size can be zero if not used)
    if vocab is None:
        raise ValueError("make_data_iter requires a `vocab` argument or dataset.vocab for numericalization")

    # Build DataLoader. You can tune num_workers if desired.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=actual_shuffle,
        collate_fn=lambda b: collate_fn(b, vocab, trg_size, lowercase=lowercase, return_lengths=True),
        pin_memory=False,
        drop_last=False,
    )

    return loader



# ------------------------
# load_data wrapper
# ------------------------
def load_data(cfg: dict) -> (Dataset, Dataset, Dataset, object, object):
    data_cfg = cfg["data"]

    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")

    train_path, dev_path, test_path = data_cfg["train"], data_cfg["dev"], data_cfg["test"]

    level = data_cfg.get("level", "word")
    lowercase = data_cfg.get("lowercase", False)
    max_sent_length = data_cfg.get("max_sent_length", None)

    tok_fun = (lambda s: list(s)) if level == "char" else (lambda s: s.split())

    trg_size = cfg["model"]["trg_size"] + 1
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'

    # build raw datasets (no torchtext Fields used)
    train_data = SignProdDataset(train_path, ("." + src_lang, "." + trg_lang, "." + files_lang), trg_size, skip_frames)
    dev_data = SignProdDataset(dev_path, ("." + src_lang, "." + trg_lang, "." + files_lang), trg_size, skip_frames)
    test_data = SignProdDataset(test_path, ("." + src_lang, "." + trg_lang, "." + files_lang), trg_size, skip_frames)

    # Optionally filter by max_sent_length if provided
    if max_sent_length is not None:
        def _filter_dataset(ds: SignProdDataset):
            filtered_src, filtered_trg, filtered_files = [], [], []
            for s, t, f in zip(ds.src, ds.trg, ds.files):
                tok_len = len(s.split())
                if tok_len <= max_sent_length and t.size(0) <= max_sent_length:
                    filtered_src.append(s)
                    filtered_trg.append(t)
                    filtered_files.append(f)
            ds.src, ds.trg, ds.files = filtered_src, filtered_trg, filtered_files

        _filter_dataset(train_data)
        _filter_dataset(dev_data)
        _filter_dataset(test_data)

    # Build source vocab using user's build_vocab helper if available
    src_max_size = data_cfg.get("src_voc_limit", None)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)

    try:
        # If the project provides a build_vocab helper that expects these args
        src_vocab = build_vocab(field="src", min_freq=src_min_freq, max_size=src_max_size or None,
                                dataset=train_data, vocab_file=src_vocab_file)
    except Exception:
        # Fallback: build simple dict-based vocab from training data
        counter = {}
        for s in train_data.src:
            for tok in s.split():
                counter[tok] = counter.get(tok, 0) + 1
        # sort and keep top-k if max_size provided
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if src_max_size:
            items = items[:src_max_size]
        stoi = {tok: i + 2 for i, (tok, _) in enumerate(items)}  # reserve 0,1 for PAD/UNK
        stoi[PAD_TOKEN] = 0
        stoi[UNK_TOKEN] = 1

        class SimpleVocab:
            def __init__(self, stoi):
                self.stoi = stoi

            def __getitem__(self, token):
                return self.stoi.get(token, self.stoi.get(UNK_TOKEN, 1))

        src_vocab = SimpleVocab(stoi)

    # create target vocab placeholder to preserve original interface
    trg_vocab = [None] * trg_size

    return train_data, dev_data, test_data, src_vocab, trg_vocab
