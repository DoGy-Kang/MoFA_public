import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Standard multi-head causal self-attention (softmax in every head).
    This is the baseline (H_sig = 0).
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)

        # lower-triangular causal mask: (1, 1, T_max, T_max)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (B, H, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v                                                  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.resid_pdrop),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block (GPT-2 style)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VanillaGPT(nn.Module):
    """GPT-2 style decoder-only transformer with softmax attention (baseline)."""

    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg.block_size

        self.transformer = nn.ModuleDict(dict(
            tok_emb=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            pos_emb=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.embd_pdrop),
            blocks=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.tok_emb.weight = self.lm_head.weight          # weight tying

        self.apply(self._init_weights)
        print(f"VanillaGPT | {self.num_params():,} parameters")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        tok = self.transformer.tok_emb(idx)
        pos = self.transformer.pos_emb(torch.arange(T, device=idx.device))
        x = self.transformer.drop(tok + pos)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                        # (B, T, V)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_t], dim=1)
        return idx
