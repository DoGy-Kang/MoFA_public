import math

import torch
import torch.nn as nn

from .vanillaGPT import MLP


class CausalSelfHahAttention(nn.Module):
    """
    MoFA causal self-attention: each head is assigned a fixed attention function.

    Heads are split into contiguous groups:
        [0, n_softmax)                     : softmax  (normalised, competitive)
        [n_softmax, n_softmax + n_sig)     : sigmoid  (independent gating,
                                              length-normalised by -log t)
        [n_softmax + n_sig, n_head)        : tanh     (experimental; unused in the paper)

    The fused QKV projection is identical to the baseline; each head uses its
    own slice. Only the score-to-weight function differs between heads.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        self.n_sig = cfg.n_sig_head
        self.n_tanh = cfg.n_tanh_head
        self.n_softmax = self.n_head - self.n_sig - self.n_tanh
        assert self.n_softmax >= 0, "n_sig_head + n_tanh_head exceeds n_head"

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
        # causal window size t per query position (query i attends to i+1 keys)
        # shape (1, 1, T_max, 1), broadcasts over (B, H, T, T)
        self.register_buffer(
            "causal_cnt",
            torch.arange(1, cfg.block_size + 1, dtype=torch.float32).view(1, 1, -1, 1),
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, T, T)

        causal = self.mask[:, :, :T, :T]                               # (1, 1, T, T)
        cnt = self.causal_cnt[:, :, :T, :]                             # (1, 1, T, 1)

        i_sig = self.n_softmax
        i_tanh = self.n_softmax + self.n_sig
        s_sm = scores[:, :i_sig]
        s_sig = scores[:, i_sig:i_tanh]
        s_th = scores[:, i_tanh:i_tanh + self.n_tanh]

        # softmax heads
        a_sm = torch.softmax(s_sm.masked_fill(causal == 0, float("-inf")), dim=-1)

        # sigmoid heads: sigma(-inf) = 0, so future positions are zeroed out
        a_sig = s_sig.masked_fill(causal == 0, float("-inf"))
        a_sig = torch.sigmoid(a_sig - cnt.log())

        # tanh heads (experimental): tanh(-inf) = -1, so mask AFTER activation
        a_th = torch.tanh(0.25 * s_th) * causal
        a_th = a_th / cnt.sqrt()

        att = torch.cat([a_sm, a_sig, a_th], dim=1)                    # (B, H, T, T)
        att = self.attn_drop(att)

        out = att @ v                                                  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class HahTransformerBlock(nn.Module):
    """Pre-LayerNorm block (GPT-2 style) with MoFA attention."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfHahAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HahGPT(nn.Module):
    """
    GPT-2 style decoder-only transformer with MoFA attention.
    Identical to `VanillaGPT` except for the per-head attention function.
    """

    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg.block_size

        self.transformer = nn.ModuleDict(dict(
            tok_emb=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            pos_emb=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.embd_pdrop),
            blocks=nn.ModuleList([HahTransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.tok_emb.weight = self.lm_head.weight          # weight tying

        self.apply(self._init_weights)
        print(f"HahGPT | {self.num_params():,} parameters")

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
