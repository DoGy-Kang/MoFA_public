"""
Train a softmax-only baseline or a MoFA GPT-2 (124M) model on OpenWebText.

Examples:
    # baseline (H_sig = 0)
    python run.py -seed 42 -d 0 -m v

    # MoFA with 3 sigmoid heads out of 12
    python run.py -seed 42 -d 0 -m h -n_sig 3
"""

import logging
import math
import os
import random
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from config import NLPConfig, hahNLPConfig
from model import HahGPT, VanillaGPT
from nlp_dataset import NLPDataset
from util import getParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="  Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    # NOTE: the model is intentionally NOT switched back to train mode here, to
    # match the behaviour used for the reported results (dropout is active only
    # until the first validation at step `steps_per_log`).
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


def make_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to zero."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, step, val_loss, val_ppl, path, config):
    torch.save({
        "epoch": step,          # field name kept for backward compatibility; stores the step
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "cfg": config,
    }, path)


def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_raw, eval_steps, step = [], [], 0
    for chunk in history["raw_train_losses"]:
        all_raw.extend(chunk)
        step += len(chunk)
        eval_steps.append(step)
    raw_steps = range(1, len(all_raw) + 1)

    axes[0].plot(raw_steps, all_raw, "b--", alpha=0.5, linewidth=0.7, label="raw step loss")
    axes[0].plot(eval_steps, history["train_loss"], "b-", label="train (EMA)")
    axes[0].plot(eval_steps, history["val_loss"], "r-", label="val")
    axes[0].set_ylim(top=max(history["train_loss"]) + 1.0)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(raw_steps, [math.exp(l) for l in all_raw], "b--", alpha=0.5,
                 linewidth=0.7, label="raw step PPL")
    axes[1].plot(eval_steps, history["train_ppl"], "b-", label="train (EMA)")
    axes[1].plot(eval_steps, history["val_ppl"], "r-", label="val")
    axes[1].set_ylim(bottom=0, top=max(history["train_ppl"]) + 10.0)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Perplexity")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"{history['mode']} — {history['datasetname'].upper()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def _quick_generate_samples(expname, model, tokenizer, CFG, step):
    """Write a few qualitative generations to disk (sanity check only)."""
    model.eval()

    prompts = [
        "Space is so fascinating that",
        "South Korea is",
        "An red apple",     # kept verbatim from the original runs
        "Sadness is not",
    ]
    save_fp = os.path.join(CFG.generate_path, expname, f"{expname}_{step}.txt")
    with open(save_fp, "w", encoding="utf-8") as f:
        f.write(f"Quick generation @ step {step}\n\n")
        for prompt in prompts:
            enc = tokenizer.encode(prompt, return_tensors="pt").to(CFG.device)
            out = model.generate(enc, max_new_tokens=150, temperature=0.8, top_k=40)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            f.write(f"Prompt: {prompt}\nOutput: {text}\n" + "-" * 40 + "\n")


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def run(expname, mode, datasetname, CFG):
    set_seed(CFG.seed)
    g = torch.Generator()
    g.manual_seed(CFG.seed)

    for d in [CFG.ckpt_path, CFG.plot_path, CFG.log_path, f"{CFG.generate_path}/{expname}"]:
        os.makedirs(d, exist_ok=True)

    log = make_logger(f"{CFG.log_path}/{expname}.log")
    log.info(f"Experiment  : {expname}")
    log.info(f"Seed        : {CFG.seed}")
    log.info(f"Device      : {CFG.device}")
    log.info(f"Layers      : {CFG.n_layer}")
    log.info(f"Embedding   : {CFG.n_embd}")
    log.info(f"Total heads : {CFG.n_head}")
    if hasattr(CFG, "n_sig_head"):
        log.info(f"Sigmoid heads: {CFG.n_sig_head} | Tanh heads: {CFG.n_tanh_head}")

    # ── tokenizer ──────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(CFG.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
    CFG.vocab_size = tokenizer.vocab_size
    log.info(f"Vocab       : {CFG.vocab_size:,}")

    # ── data ───────────────────────────────────
    raw_train = load_from_disk(CFG.train_data_dir)
    raw_val = load_from_disk(CFG.val_data_dir)
    train_dataset = NLPDataset(raw_train, tokenizer, CFG.block_size,
                               cache_path=f"{CFG.cache_dir}/train_tokens.pt")
    val_dataset = NLPDataset(raw_val, tokenizer, CFG.block_size,
                             cache_path=f"{CFG.cache_dir}/val_tokens.pt")

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True,
        num_workers=4, worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True,
        num_workers=4,
    )
    log.info(f"Train batches : {len(train_loader):,}")
    log.info(f"Val   batches : {len(val_loader):,}")

    # ── model / optimizer ──────────────────────
    if mode == "vanilla_gpt":
        model = VanillaGPT(CFG).to(CFG.device)
    elif mode == "hah_gpt":
        model = HahGPT(CFG).to(CFG.device)
    else:
        raise ValueError(f"Unknown model mode: {mode}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, betas=(0.9, 0.95),
    )
    total_steps = CFG.total_steps
    scheduler = make_scheduler(optimizer, CFG.warmup_steps, total_steps)
    eval_interval_steps = CFG.steps_per_log

    history = dict(mode=mode, datasetname=datasetname, train_loss=[], train_ppl=[],
                   raw_train_losses=[], val_loss=[], val_ppl=[])
    best_val_ppl, best_step = float("inf"), -1

    log.info(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Total steps : {total_steps} | Eval every: {eval_interval_steps}")
    log.info("=" * 60)

    train_iter = iter(train_loader)     # recycled when exhausted
    ema_decay, ema_train_loss = 0.98, None
    raw_losses = []

    pbar = trange(1, total_steps + 1, desc="train")
    for step in pbar:
        optimizer.zero_grad()
        accumulated_loss = 0.0
        for _ in range(CFG.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(CFG.device), y.to(CFG.device)
            _, loss = model(x, y)
            loss = loss / CFG.grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optimizer.step()
        scheduler.step()

        raw_losses.append(accumulated_loss)
        ema_train_loss = (accumulated_loss if ema_train_loss is None
                          else ema_decay * ema_train_loss + (1 - ema_decay) * accumulated_loss)

        pbar.set_postfix(loss=f"{accumulated_loss:.4f}", ema=f"{ema_train_loss:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # ── periodic validation / checkpoint ───
        if step % eval_interval_steps == 0 or step == total_steps:
            raw_train_loss = sum(raw_losses) / len(raw_losses)
            val_loss, val_ppl = evaluate(model, val_loader, CFG.device)

            history["train_loss"].append(ema_train_loss)
            history["train_ppl"].append(math.exp(ema_train_loss))
            history["raw_train_losses"].append(list(raw_losses))
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            raw_losses = []

            if val_ppl < best_val_ppl:
                best_step, best_val_ppl = step, val_ppl
                save_checkpoint(model, optimizer, step, val_loss, val_ppl,
                                f"{CFG.ckpt_path}/{expname}.pth", CFG)

            log.info(
                f"step {step:>6d}/{total_steps} | "
                f"train {ema_train_loss:.4f} (raw {raw_train_loss:.4f}) | "
                f"val {val_loss:.4f} ppl {val_ppl:.2f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | best step {best_step}"
            )
            plot_curves(history, save_path=f"{CFG.plot_path}/{expname}.png")

        if (step % (10 * eval_interval_steps) == 0 or step == total_steps
                or step in [125, 250, 500, 1000, 2000, 4000]):
            _quick_generate_samples(expname, model, tokenizer, CFG, step)


if __name__ == "__main__":
    st = dt.now()
    args = getParser()
    dataset_name = "openwebtext"
    mode = {"v": "vanilla_gpt", "h": "hah_gpt"}[args.mode]
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    if mode == "vanilla_gpt":
        CFG = NLPConfig(seed=args.seed, device=device)
        expname = f"{dataset_name}_seed{args.seed}_pure_{mode}_{st.strftime('%Y%m%d_%H%M')}"
    else:
        CFG = hahNLPConfig(seed=args.seed, n_sig_head=args.n_sig_head,
                           n_tanh_head=args.n_tanh_head, device=device)
        expname = (f"{dataset_name}_seed{args.seed}_pure_{mode}"
                   f"_sig{CFG.n_sig_head}_tanh{CFG.n_tanh_head}_{st.strftime('%Y%m%d_%H%M')}")

    if args.total_steps is not None:
        CFG.total_steps = args.total_steps

    run(expname, mode, dataset_name, CFG)
    print(f"Running time: {dt.now() - st}")
