import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import FeedForward, MultiHeadAttention, PositionalEncoding


class GPTBlock(nn.Module):
    """
    GPT-style block:
    1. causal self-attention
    2. feed-forward

    No cross-attention because there is no encoder.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask=causal_mask
        )
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    GPT-style model.

    Flow:
    tokens -> causal blocks -> logits for next token prediction
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embed.weight

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape
        causal_mask = self._causal_mask(seq_len, x.device)

        h = self.pos_enc(self.token_embed(x) * math.sqrt(self.d_model))
        for block in self.blocks:
            h = block(h, causal_mask)

        h = self.norm(h)
        return self.output_projection(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        seq = prompt.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(seq)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

        return seq
