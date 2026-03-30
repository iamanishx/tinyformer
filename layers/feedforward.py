"""
Feed-Forward Network
====================

THE SIMPLE PART:
  Two linear layers with ReLU in between.
  Applied independently to each token position.

  FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

WHY IT EXISTS:
  Self-attention mixes information BETWEEN tokens.
  FFN processes information WITHIN each token.

  Together they give the model:
    - Attention = communication between positions
    - FFN = computation within each position

WHY d_ff = 4 * d_model:
  Standard practice. Gives enough capacity for the non-linearity
  to learn useful transformations.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Same architecture used in encoder, decoder, and decoder-only models.

    Args:
        d_model: input and output dimension
        d_ff: hidden dimension (typically 4 * d_model)
        dropout: dropout rate after first layer
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.net(x)
