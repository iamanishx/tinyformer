"""
Shared Layers Package
=====================
  PositionalEncoding  → adds position info to embeddings
  MultiHeadAttention  → the core attention mechanism
  FeedForward         → position-wise transformation
"""

from .attention import MultiHeadAttention
from .embedding import PositionalEncoding
from .feedforward import FeedForward

__all__ = ["MultiHeadAttention", "PositionalEncoding", "FeedForward"]
