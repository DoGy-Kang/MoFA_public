import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class NLPDataset(Dataset):
    """
    OpenWebText language-modeling dataset.

    Every document is suffixed with the EOS token (<|endoftext|>) and all documents
    are concatenated into one token stream, which is cut into non-overlapping
    windows of `block_size` tokens (inputs) shifted by one (targets).
    The tokenized stream is cached to `cache_path` on first use.
    """

    def __init__(self, hf_dataset, tokenizer, block_size, cache_path=None):
        self.block_size = block_size

        if cache_path and os.path.exists(cache_path):
            print(f"Loading tokenized cache from {cache_path} ...")
            self.data = torch.load(cache_path)
            print(f"Total tokens: {len(self.data):,}")
            return

        print("Tokenizing ...")
        all_ids = []
        for example in tqdm(hf_dataset):
            all_ids.extend(tokenizer.encode(example["text"] + tokenizer.eos_token,
                                            add_special_tokens=False))
        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"Total tokens: {len(self.data):,}")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.data, cache_path)
            print(f"Saved cache to {cache_path}")

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start:start + self.block_size + 1]
        return chunk[:-1], chunk[1:]
