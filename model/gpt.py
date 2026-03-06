# ============================================================
# model/gpt.py — MiniLLM Transformer
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv        = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out        = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim)
                    .transpose(1, 2) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        attn = attn.masked_fill(causal, float('-inf'))

        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln1    = nn.LayerNorm(embed_dim)
        self.attn   = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2    = nn.LayerNorm(embed_dim)
        self.ff     = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads,
                 num_layers, ffn_dim, max_seq_len, dropout=0.1):
        super().__init__()

        self.token_emb  = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb    = nn.Embedding(max_seq_len, embed_dim)
        self.dropout    = nn.Dropout(dropout)

        self.blocks     = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f       = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying — token embedding and output head share weights
        self.head.weight = self.token_emb.weight

        # Init weights
        self.apply(self._init_weights)

        total = sum(p.numel() for p in self.parameters())
        print(f"[MiniLLM] Parameters: {total:,}  ({total/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x   = self.dropout(tok + pos)

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss


def build_model(config):
    model = MiniLLM(
        vocab_size  = config.VOCAB_SIZE,
        embed_dim   = config.EMBED_DIM,
        num_heads   = config.NUM_HEADS,
        num_layers  = config.NUM_LAYERS,
        ffn_dim     = config.FFN_DIM,
        max_seq_len = config.MAX_SEQ_LEN,
        dropout     = config.DROPOUT,
    )
    return model.to(config.DEVICE)
