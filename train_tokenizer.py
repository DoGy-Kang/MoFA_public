"""
Train a GPT-2-style byte-level BPE tokenizer from scratch on OpenWebText.

Run `prepare_data.py` first.

Usage:
    python train_tokenizer.py
"""

import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast


def openweb_tokenizer(vocab_size=50257, data_fraction=0.2):
    output_dir = f"./tokenizer_{vocab_size // 1000}k_openweb"
    os.makedirs(output_dir, exist_ok=True)

    # 1) First `data_fraction` of the training split
    ds = load_from_disk("./data/openwebtext_arrow/train")
    ds = ds.select(range(int(len(ds) * data_fraction)))
    print(f"Training tokenizer on {len(ds):,} documents ({data_fraction * 100:.0f}%)")

    # 2) Byte-level BPE
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # GPT-2 style
    tok.decoder = decoders.ByteLevel()

    # 3) Special tokens; <|endoftext|> serves as BOS / EOS / document separator
    special_tokens = ["<unk>", "<pad>", "<|endoftext|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
    )

    # 4) Append a document-boundary marker to each document
    def batch_iterator(batch_size=1000):
        batch = []
        for ex in ds:
            txt = ex.get("text", None)
            if txt:
                batch.append(txt + "<|endoftext|>")
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    tok.train_from_iterator(batch_iterator(), trainer=trainer)

    # 5) Wrap and save as a Hugging Face tokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tok.save_pretrained(output_dir)
    print(f"Saved tokenizer (vocab {vocab_size}) to '{output_dir}'")
    return output_dir


if __name__ == "__main__":
    openweb_tokenizer()
