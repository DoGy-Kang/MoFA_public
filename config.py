"""
Experiment configurations.

NOTE: class names (`NLPConfig`, `hahNLPConfig`) are kept as-is because the
config object is pickled into every checkpoint; renaming them would break
loading of released checkpoints.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class NLPConfig:
    """Baseline (softmax-only) GPT-2 124M configuration."""

    # --- model ---
    vocab_size: int = 50257          # overwritten by the tokenizer at runtime
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1

    # --- training ---
    batch_size: int = 32             # micro-batch size
    grad_accum_steps: int = 16       # effective batch = 32 x 16 sequences x 1024 tokens
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    total_steps: int = 50000
    steps_per_log: int = 500         # validation / checkpoint interval

    # --- data ---
    tokenizer_dir: str = "./tokenizer_50k_openweb"
    train_data_dir: str = "./data/openwebtext_arrow/train"
    val_data_dir: str = "./data/openwebtext_arrow/val"
    cache_dir: str = "./data/cache"

    # --- misc ---
    seed: int = 42
    ckpt_path: str = "./Results_openweb/checkpoint"
    plot_path: str = "./Results_openweb/plot"
    log_path: str = "./Results_openweb/log"
    generate_path: str = "./Results_openweb/generate"
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class hahNLPConfig(NLPConfig):
    """MoFA configuration: heads are split into contiguous softmax / sigmoid groups.

    Heads [0, n_head - n_sig_head - n_tanh_head) use softmax,
    the next n_sig_head heads use length-normalised sigmoid.
    `n_tanh_head` is an experimental option and is 0 in all reported experiments.
    """

    n_sig_head: int = 3
    n_tanh_head: int = 0
