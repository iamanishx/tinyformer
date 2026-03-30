import math

import torch
import torch.nn as nn

from layers import FeedForward, MultiHeadAttention, PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Encoder block:
    1. bidirectional self-attention
    2. feed-forward
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    """
    Decoder block:
    1. masked self-attention
    2. cross-attention to encoder output
    3. feed-forward
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        masked_attn = self.masked_self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(masked_attn))

        cross_attn = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class EncoderDecoderTransformer(nn.Module):
    """
    Full seq2seq transformer.

    Flow:
    source tokens -> encoder -> encoder representations
    target tokens -> decoder + cross-attn to encoder reps -> logits
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        _, tgt_len = tgt.shape
        tgt_mask = self._causal_mask(tgt_len, tgt.device)

        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm_dec(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask)
        return self.output_projection(dec_output)
