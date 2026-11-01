"""
Download OpenWebText and create a fixed 99/1 train/validation split.

Usage:
    python prepare_data.py
"""

from datasets import load_dataset


def download_data(split_seed=42, val_fraction=0.01):
    ds_full = load_dataset("openwebtext", cache_dir="./data/openwebtext")

    split_ds = ds_full["train"].train_test_split(test_size=val_fraction, seed=split_seed)

    split_ds["train"].save_to_disk("./data/openwebtext_arrow/train")
    split_ds["test"].save_to_disk("./data/openwebtext_arrow/val")
    print("Saved splits to ./data/openwebtext_arrow/{train,val}")


if __name__ == "__main__":
    download_data()
