"""
Multi-Head Attention
====================

THE CORE MECHANISM:
  Attention(query, key, value) = softmax(QK^T / sqrt(d)) @ V

  In plain English:
    "For each query, find which keys are relevant,
     then return a weighted sum of their values"

WHY MULTI-HEAD:
  One attention head = one "pattern" of relationships
  Multi-head = multiple patterns in parallel

  Head 1 might learn: subject-verb agreement
  Head 2 might learn: pronoun references
  Head 3 might learn: positional patterns
  etc.

HOW IT WORKS:
  1. Project input into Q, K, V (same input or different)
  2. Split into multiple heads
  3. Each head computes attention independently
  4. Concatenate all heads
  5. Final linear projection

USED IN THREE PLACES:

  1. ENCODER SELF-ATTENTION:
     Q, K, V all from encoder input
     No mask → every token sees every token

  2. DECODER MASKED SELF-ATTENTION:
     Q, K, V all from decoder input
     Causal mask → each token sees only itself and past

  3. DECODER CROSS-ATTENTION:
     Q from decoder, K and V from encoder output
     This is how decoder reads the "recipe" from encoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention mechanism.

    Args:
        d_model: model dimension (must be divisible by num_heads)
        num_heads: number of attention heads
        dropout: dropout rate on attention weights
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  (seq_len_q, seq_len_k) or broadcastable
                   0 = attend normally
                   -inf = do not attend (softmax → 0)

        Returns:
            (batch, seq_len_q, d_model)
        """
        B, T_q, _ = query.shape
        T_k = key.size(1)

        Q = self.W_q(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        return self.W_o(out)
