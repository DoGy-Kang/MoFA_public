"""
Zero-shot perplexity evaluation on 15 out-of-distribution domains.

Examples:
    # evaluate every checkpoint in a directory
    python validate.py --ckpt_dir ./Results_openweb/checkpoint

    # evaluate specific checkpoints
    python validate.py --ckpt_dir ./Results_openweb/checkpoint \
        --ckpts openwebtext_seed42_pure_hah_gpt_sig3_tanh0_20260601_1200.pth

Evaluation protocol (used for all reported results):
    * Texts of a domain are (optionally grouped and) tokenized into one continuous
      token stream, without document separators.
    * The stream is cut into overlapping windows of `block_size` (1024) tokens with
      stride = min_chunk_size // 4 (= 128); windows shorter than min_chunk_size
      (= 512) are dropped, the remaining short tail windows are right-padded.
    * Perplexity = exp(mean of per-batch mean token cross-entropy).
    * At most `num_samples` (= 10,000) texts per domain are used.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────
class TextDataset(Dataset):
    """Concatenate all texts into one token stream and cut overlapping windows."""

    def __init__(self, texts, tokenizer, block_size=1024, max_samples=None,
                 min_chunk_ratio=0.5):
        self.samples = []
        self.block_size = block_size
        self.min_chunk_size = int(block_size * min_chunk_ratio)

        texts = texts[:max_samples] if max_samples else texts

        all_tokens, skipped = [], 0
        for text in tqdm(texts, desc="Tokenizing", leave=False):
            if not text or len(text.strip()) < 3:
                skipped += 1
                continue
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if tokens:
                    all_tokens.extend(tokens)
            except Exception:
                skipped += 1

        print(f"  Total tokens: {len(all_tokens)} | texts skipped: {skipped}")

        if not all_tokens:
            self.stats = {"total_texts": len(texts), "skipped": skipped,
                          "chunks_created": 0, "strategy": "none (no valid text)"}
            return

        stride = (self.min_chunk_size // 4 if self.min_chunk_size > 64
                  else max(1, self.min_chunk_size // 2))
        self.samples = _make_windows(all_tokens, self.block_size, self.min_chunk_size,
                                     stride, tokenizer.pad_token_id)

        n_valid = len(texts) - skipped
        self.stats = {
            "total_texts": len(texts),
            "skipped": skipped,
            "total_tokens": len(all_tokens),
            "chunks_created": len(self.samples),
            "avg_tokens_per_text": len(all_tokens) / n_valid if n_valid > 0 else 0,
            "strategy": "concatenated tokens with stride",
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample[:-1], sample[1:]


class SmartTextDataset(Dataset):
    """Group short texts (joined by a space) before tokenizing, then cut windows."""

    def __init__(self, texts, tokenizer, block_size=1024, max_samples=None,
                 min_chunk_ratio=0.5, group_size=50):
        self.samples = []
        self.block_size = block_size
        self.min_chunk_size = int(block_size * min_chunk_ratio)

        texts = texts[:max_samples] if max_samples else texts

        # group texts until a group reaches min_chunk_size tokens or group_size texts
        grouped_texts, current_group, group_token_count = [], [], 0
        for text in texts:
            if not text or len(text.strip()) < 3:
                continue
            try:
                group_token_count += len(tokenizer.encode(text, add_special_tokens=False))
                current_group.append(text)
                if group_token_count >= self.min_chunk_size or len(current_group) >= group_size:
                    grouped_texts.append(" ".join(current_group))
                    current_group, group_token_count = [], 0
            except Exception:
                continue
        if current_group:
            grouped_texts.append(" ".join(current_group))
        print(f"  Created {len(grouped_texts)} groups from {len(texts)} texts")

        all_tokens = []
        for text in tqdm(grouped_texts, desc="Tokenizing groups", leave=False):
            try:
                all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                continue
        print(f"  Total tokens: {len(all_tokens)}")

        stride = max(self.min_chunk_size // 4, 1)
        self.samples = _make_windows(all_tokens, self.block_size, self.min_chunk_size,
                                     stride, tokenizer.pad_token_id)

        self.stats = {
            "total_texts": len(texts),
            "grouped_texts": len(grouped_texts),
            "total_tokens": len(all_tokens),
            "chunks_created": len(self.samples),
            "strategy": "grouped short texts then concatenated",
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample[:-1], sample[1:]


def _make_windows(all_tokens, block_size, min_chunk_size, stride, pad_token_id):
    """Overlapping (block_size + 1)-token windows; short tail windows are right-padded."""
    samples = []
    for i in range(0, len(all_tokens) - min_chunk_size, stride):
        chunk = all_tokens[i:i + block_size + 1]
        if len(chunk) >= min_chunk_size + 1:
            if len(chunk) < block_size + 1:
                chunk = chunk + [pad_token_id] * (block_size + 1 - len(chunk))
            samples.append(torch.tensor(chunk[:block_size + 1], dtype=torch.long))
    return samples


# ─────────────────────────────────────────────
# Out-of-distribution domains
#   * splits are 'test' wherever available
#   * bookcorpus / reddit_tifu only ship a train split -> sliced
#   * ccdv/* versions replace the broken scientific_papers loader
# ─────────────────────────────────────────────
DATASETS_TO_EVAL = {
    "code":       {"name": "code_search_net", "config": "python", "split": "test",
                   "text_column": "func_code_string", "kwargs": {"trust_remote_code": True}},
    "novel":      {"name": "bookcorpus", "split": "train[:5%]",
                   "text_column": "text", "kwargs": {"trust_remote_code": True}},
    "news":       {"name": "cnn_dailymail", "config": "3.0.0", "split": "test",
                   "combine_columns": ["article", "highlights"], "kwargs": {}},
    "pubmed":     {"name": "ccdv/pubmed-summarization", "config": "document", "split": "test",
                   "text_column": "abstract", "kwargs": {"trust_remote_code": True}},
    "arxiv":      {"name": "ccdv/arxiv-summarization", "config": "document", "split": "test",
                   "text_column": "abstract", "kwargs": {"trust_remote_code": True}},
    "wikitext":   {"name": "wikitext", "config": "wikitext-103-v1", "split": "test",
                   "text_column": "text", "kwargs": {}},
    "reddit":     {"name": "reddit_tifu", "config": "long", "split": "train[:5%]",
                   "text_column": "documents", "kwargs": {"trust_remote_code": True}},
    "legal":      {"name": "billsum", "split": "test",
                   "text_column": "text", "kwargs": {}},
    "dialogue":   {"name": "roskoN/dailydialog", "split": "test",
                   "text_column": "utterances", "kwargs": {}},          # List[str], joined
    "empathetic": {"name": "facebook/empathetic_dialogues", "split": "test",
                   "text_column": "utterance", "kwargs": {"trust_remote_code": True}},
    "tweets":     {"name": "cardiffnlp/tweet_eval", "config": "emotion", "split": "test",
                   "text_column": "text", "kwargs": {}},
    "sst2":       {"name": "stanfordnlp/sst2", "split": "validation",
                   "text_column": "sentence", "kwargs": {}},
    "lambada":    {"name": "EleutherAI/lambada_openai", "config": "en", "split": "test",
                   "text_column": "text", "kwargs": {}},
    "math":       {"name": "gsm8k", "config": "main", "split": "test",
                   "combine_columns": ["question", "answer"], "kwargs": {}},
    "ptb":        {"name": "ptb_text_only", "split": "test",
                   "text_column": "sentence", "kwargs": {"trust_remote_code": True}},
}

# short-sentence domains: group more texts together
SHORT_TEXT_DATASETS = {"sst2", "tweets", "dialogue", "empathetic", "ptb"}


def load_texts(config_dict):
    if "config" in config_dict:
        dataset = load_dataset(config_dict["name"], config_dict["config"],
                               split=config_dict["split"], **config_dict["kwargs"])
    else:
        dataset = load_dataset(config_dict["name"], split=config_dict["split"],
                               **config_dict["kwargs"])

    if "combine_columns" in config_dict:
        cols = [dataset[c] for c in config_dict["combine_columns"]]
        texts = [" ".join(str(x).strip() for x in row if x) for row in zip(*cols)]
    else:
        raw_texts = dataset[config_dict["text_column"]]
        if len(raw_texts) > 0 and isinstance(raw_texts[0], list):
            texts = [" ".join(str(x) for x in t) for t in raw_texts]
        else:
            texts = raw_texts

    return [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataset, device, batch_size=32):
    model.eval()
    if len(dataset) == 0:
        return None, None, {"error": "Empty dataset"}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss, num_batches = 0.0, 0
    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return None, None, {"error": "No batches processed"}

    avg_loss = total_loss / num_batches
    return avg_loss, math.exp(avg_loss), {"num_batches": num_batches,
                                           "total_loss": float(total_loss)}


def validate(exp_name, checkpoint_path, tokenizer_dir="./tokenizer_50k_openweb",
             device="cuda:0", batch_size=128, num_samples=None, block_size=1024,
             dataset_strategy="smart", out_dir="./External_Results"):
    """
    Args:
        dataset_strategy: 'smart' (SmartTextDataset, used in the paper) or
                          'concat' (TextDataset)
    """
    from model import HahGPT, VanillaGPT

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["cfg"]
    model = (HahGPT(config) if hasattr(config, "n_sig_head") else VanillaGPT(config)).to(device)
    model.load_state_dict(checkpoint["model"])

    results = {"checkpoint": checkpoint_path, "block_size": block_size,
               "dataset_strategy": dataset_strategy, "datasets": {}}

    for name, config_dict in DATASETS_TO_EVAL.items():
        try:
            print(f"\n▶ {name.upper()}")
            texts = load_texts(config_dict)
            print(f"  Loaded {len(texts)} texts")

            if dataset_strategy == "smart":
                text_dataset = SmartTextDataset(
                    texts, tokenizer, block_size=block_size, max_samples=num_samples,
                    min_chunk_ratio=0.5,
                    group_size=200 if name in SHORT_TEXT_DATASETS else 50,
                )
            else:
                text_dataset = TextDataset(
                    texts, tokenizer, block_size=block_size, max_samples=num_samples,
                    min_chunk_ratio=0.5,
                )

            if len(text_dataset) == 0:
                print("  No valid chunks created")
                results["datasets"][name] = {"error": "No valid chunks created",
                                             "stats": text_dataset.stats}
                continue

            loss, ppl, eval_stats = evaluate(model, text_dataset, device, batch_size)
            if loss is None:
                results["datasets"][name] = {"error": eval_stats["error"],
                                             "stats": text_dataset.stats}
                print(f"  {eval_stats['error']}")
            else:
                results["datasets"][name] = {"loss": float(loss), "perplexity": float(ppl),
                                             "eval_stats": eval_stats,
                                             "dataset_stats": text_dataset.stats}
                print(f"  Loss: {loss:.4f} | PPL: {ppl:.2f}")

        except Exception as e:
            import traceback
            print(f"  Failed: {e}")
            results["datasets"][name] = {"error": str(e)}
            traceback.print_exc()

    ppls = [m["perplexity"] for m in results["datasets"].values() if "perplexity" in m]
    if ppls:
        results["summary"] = {
            "mean_ppl": float(np.mean(ppls)), "std_ppl": float(np.std(ppls)),
            "min_ppl": float(np.min(ppls)), "max_ppl": float(np.max(ppls)),
            "num_successful": len(ppls),
            "num_failed": len(results["datasets"]) - len(ppls),
        }
        print(f"\nSuccessful: {len(ppls)} / {len(results['datasets'])} | "
              f"mean PPL {results['summary']['mean_ppl']:.2f}")

    os.makedirs(out_dir, exist_ok=True)
    results_path = f"{out_dir}/validation_results_{exp_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    return results


def exp_name_from_ckpt(filename):
    """
    'openwebtext_seed42_pure_hah_gpt_sig3_tanh0_<date>_<time>.pth' -> 'hah_sig3_seed42'
    'openwebtext_seed42_pure_vanilla_gpt_<date>_<time>.pth'       -> 'vanilla_seed42'
    """
    parts = filename.split("_")
    seed, name = parts[1], parts[3]
    if name == "hah":
        name += "_" + parts[5]
    return f"{name}_{seed}"


def get_args():
    parser = argparse.ArgumentParser(description="Zero-shot OOD perplexity evaluation.")
    parser.add_argument("--ckpt_dir", type=str, default="./Results_openweb/checkpoint")
    parser.add_argument("--ckpts", type=str, nargs="*", default=None,
                        help="Checkpoint file names inside --ckpt_dir (default: all .pth)")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer_50k_openweb")
    parser.add_argument("-d", "--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Max number of texts per domain")
    parser.add_argument("--strategy", choices=["smart", "concat"], default="smart")
    parser.add_argument("--out_dir", type=str, default="./External_Results")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    ckpts = args.ckpts or sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith(".pth"))

    for ckpt in ckpts:
        validate(
            exp_name_from_ckpt(ckpt),
            os.path.join(args.ckpt_dir, ckpt),
            tokenizer_dir=args.tokenizer_dir,
            device=device,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            dataset_strategy=args.strategy,
            out_dir=args.out_dir,
        )
