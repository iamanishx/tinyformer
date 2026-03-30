"""
Positional Encoding
===================

THE PROBLEM:
  Transformers have no notion of "position".
  The sentence "man bites dog" and "dog bites man"
  would have identical representations.

  RNNs: position is implicit (process left-to-right)
  CNNs: position is implicit (sliding window moves)
  Transformers: position must be INJECTED explicitly.

THE SOLUTION:
  Add a positional encoding to each token embedding.

  Input to model = TokenEmbedding + PositionalEncoding

THE CLASSIC APPROACH (sine/cosine):
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

WHY SIN/COS:
  1. Each position gets a UNIQUE encoding
  2. Nearby positions have SIMILAR encodings (smooth)
  3. Model can learn RELATIVE positions
     (sin(a+b) can be expressed as linear combination of sin(a) and cos(a))
  4. Extrapolates to longer sequences than seen in training

ALTERNATIVES IN MODERN LLMS:
  - Learned embeddings (fixed max length, trained)
  - RoPE (rotary - used in LLaMA, GPT-NeoX)
  - ALiBi (attention bias - used in some models)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    Adds position information to token embeddings.

    Args:
        d_model: embedding dimension
        max_len: maximum sequence length to support
        dropout: dropout rate after adding PE
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) - token embeddings
        Returns:
            (batch, seq_len, d_model) - embeddings + positional encoding
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
