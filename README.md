# Attention Function as an Intrinsic Inductive Bias

Official code for **"Attention Function as an Intrinsic Inductive Bias: How Models' Behavior Diverges in Novel Contexts"** (NeurIPS 2026 Workshop on Developmental Perspectives on AI).

**MoFA** (Mixture of Function Attention) assigns a fixed attention function to each head of a standard GPT-2 model: some heads use softmax and the rest use length-normalised sigmoid. The ratio is set before training and never changes. We find that models with different ratios are nearly indistinguishable in-distribution but diverge systematically on zero-shot out-of-distribution text.

## Method

For a model with `H = 12` heads and `H_sig` sigmoid heads, heads are split into contiguous groups:

| Heads | Attention function |
|---|---|
| `0 … H − H_sig − 1` | `softmax(QKᵀ/√d)` |
| `H − H_sig … H − 1` | `sigmoid(QKᵀ/√d − log t)`, where `t` is the causal window size |

All other components are identical to the softmax-only baseline: the fused QKV projection, the MLP, LayerNorm, initialisation and data order.

## Repository structure

```
.
├── config.py            # model / training configurations
├── util.py              # command-line arguments
├── prepare_data.py      # download OpenWebText, create 99/1 train/val split
├── train_tokenizer.py   # train the byte-level BPE tokenizer from scratch
├── nlp_dataset.py       # OpenWebText token dataset
├── run.py               # training
├── validate.py          # zero-shot OOD perplexity on 15 domains
├── model/
│   ├── vanillaGPT.py    # softmax-only baseline
│   └── hahGPT.py        # MoFA (mixed softmax / sigmoid heads)
└── scripts/
    └── run_all.sh       # 5 ratios x 5 seeds
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Data and tokenizer**

```bash
python prepare_data.py       # -> ./data/openwebtext_arrow/{train,val}
python train_tokenizer.py    # -> ./tokenizer_50k_openweb
```

**2. Training**

```bash
# softmax-only baseline (H_sig = 0)
python run.py -seed 42 -d 0 -m v

# MoFA with H_sig sigmoid heads (H_sig in {3, 6, 9, 12})
python run.py -seed 42 -d 0 -m h -n_sig 3

# full grid used in the paper (5 ratios x 5 seeds)
bash scripts/run_all.sh 0
```

| Argument | Description |
|---|---|
| `-seed` | random seed (paper: 42, 1005, 1111, 2026, 9999) |
| `-d` | CUDA device index |
| `-m` | `v` = baseline, `h` = MoFA |
| `-n_sig` | number of sigmoid heads (out of 12) |
| `--total_steps` | optional override of the number of training steps |

Checkpoints, logs and loss curves are written to `./Results_openweb/`. The checkpoint with the lowest validation perplexity is kept.

**3. Zero-shot OOD evaluation**

```bash
# all checkpoints in a directory
python validate.py --ckpt_dir ./Results_openweb/checkpoint

# specific checkpoints
python validate.py --ckpt_dir ./Results_openweb/checkpoint --ckpts <file1>.pth <file2>.pth
```

Results are saved as JSON to `./External_Results/`.

## Training configuration

| | |
|---|---|
| Architecture | GPT-2 124M (12 layers, 12 heads, d = 768, context 1024), pre-LN, weight tying |
| Tokenizer | byte-level BPE trained on 20% of the OpenWebText training split |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1, grad clip 1.0 |
| Schedule | peak LR 3e-4, 2,000 linear warmup steps, cosine decay to 0 |
| Batch | 32 sequences × 16 grad-accumulation steps × 1024 tokens (~0.5M tokens / step) |
| Steps | 50,000 (validation every 500 steps) |
| Training data | documents joined with `<\|endoftext\|>`, non-overlapping 1024-token windows, shuffled with a seeded generator |
| Dropout | 0.1 (embedding, attention, residual). The model is left in eval mode after the first validation at step 500, so dropout is effectively active only for the first 500 steps |

## OOD evaluation protocol

| Domain | Hugging Face dataset | Split | Field |
|---|---|---|---|
| Code | `code_search_net` (python) | test | `func_code_string` |
| Novel | `bookcorpus` | train[:5%] | `text` |
| News | `cnn_dailymail` (3.0.0) | test | `article` + `highlights` |
| PubMed | `ccdv/pubmed-summarization` | test | `abstract` |
| arXiv | `ccdv/arxiv-summarization` | test | `abstract` |
| WikiText | `wikitext` (103-v1) | test | `text` |
| Reddit | `reddit_tifu` (long) | train[:5%] | `documents` |
| Legal | `billsum` | test | `text` |
| Dialogue | `roskoN/dailydialog` | test | `utterances` |
| Empathetic | `facebook/empathetic_dialogues` | test | `utterance` |
| Tweets | `cardiffnlp/tweet_eval` (emotion) | test | `text` |
| SST-2 | `stanfordnlp/sst2` | validation | `sentence` |
| LAMBADA | `EleutherAI/lambada_openai` (en) | test | `text` |
| Math | `gsm8k` (main) | test | `question` + `answer` |
| PTB | `ptb_text_only` | test | `sentence` |

- Up to 10,000 texts per domain are used.
- Texts are grouped (50 per group, or 200 for short-sentence domains) and joined by a space, then tokenized into one continuous stream.
- The stream is cut into overlapping 1024-token windows with stride 128. Windows shorter than 512 tokens are dropped.
- Perplexity is `exp` of the mean per-batch cross-entropy.
- The few tail windows are right-padded and the pad tokens are not masked from the loss. This applies identically to every configuration, so comparisons across ratios are unaffected.

## Citation

```bibtex
@inproceedings{kang2026attention,
  title     = {Attention Function as an Intrinsic Inductive Bias: How Models' Behavior Diverges in Novel Contexts},
  author    = {Kang, Dong Gyun and others},
  booktitle = {NeurIPS 2026 Workshop on Developmental Perspectives on AI},
  year      = {2026}
}
```
