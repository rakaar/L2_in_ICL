"""PyTorch transformer model mirroring the JAX Haiku stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .resnet import SimpleResNet


@dataclass
class EmbeddingConfig:
  """Configuration for the input embedder."""

  num_classes: int
  emb_dim: int
  example_encoding: str = "resnet"  # 'resnet' | 'linear' | 'embedding'
  flatten_superpixels: bool = False
  example_dropout_prob: float = 0.0
  concatenate_labels: bool = False
  use_positional_encodings: bool = True
  positional_dropout_prob: float = 0.0
  input_channels: int = 1


@dataclass
class TransformerConfig:
  """Configuration for the transformer tower."""

  num_layers: int
  num_heads: int
  dropout_prob: float = 0.0
  widening_factor: int = 4
  self_att_init_scale: float = 1.0
  dense_init_scale: float = 1.0


class SinusoidalPositionalEncoding(nn.Module):
  """Sinusoidal encoding with optional dropout."""

  def __init__(self, dropout: float = 0.0, max_time: float = 10000.0):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_time = max_time

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, dim = x.shape
    if dim % 2 != 0:
      raise ValueError("Embedding sizes must be even when using positional encodings.")
    position = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=x.dtype, device=x.device)
        * (-math.log(self.max_time) / dim)
    )
    embeddings = torch.zeros(seq_len, dim, dtype=x.dtype, device=x.device)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    embeddings = embeddings.unsqueeze(0).expand(batch, -1, -1)
    return self.dropout(embeddings)


class InputEmbedder(nn.Module):
  """Port of the Haiku input embedder."""

  def __init__(self, config: EmbeddingConfig):
    super().__init__()
    self.config = config
    self.emb_dim = config.emb_dim
    encoding = config.example_encoding.lower()
    self.resnet_projection: nn.Module | None = None
    if encoding == "linear":
      self.example_encoder = nn.Linear(105 * 105, self.emb_dim)
    elif encoding == "embedding":
      self.example_encoder = nn.Embedding(config.num_classes, self.emb_dim)
    elif encoding == "resnet":
      resnet_emb_dim = self.emb_dim
      if config.flatten_superpixels:
        resnet_emb_dim = max(1, self.emb_dim // 16)
      channels_per_group = (16, 32, 32, resnet_emb_dim)
      self.example_encoder = SimpleResNet(
          input_channels=config.input_channels,
          blocks_per_group=(2, 2, 2, 2),
          channels_per_group=channels_per_group,
          strides=(1, 2, 2, 2),
          use_projection=(True, True, True, True),
          bottleneck=True,
          flatten_superpixels=config.flatten_superpixels,
      )
      with torch.no_grad():
        was_training = self.example_encoder.training
        self.example_encoder.train(False)
        dummy = torch.zeros(
            1, config.input_channels, 105, 105, dtype=torch.float32
        )
        inferred_dim = self.example_encoder(dummy).shape[-1]
        self.example_encoder.train(was_training)
      if inferred_dim != self.emb_dim:
        self.resnet_projection = nn.Linear(inferred_dim, self.emb_dim)
    else:
      raise ValueError(f"Unsupported example_encoding '{config.example_encoding}'")

    label_vocab = config.num_classes + (1 if config.concatenate_labels else 0)
    self.label_embedding = nn.Embedding(label_vocab, self.emb_dim)
    if config.use_positional_encodings:
      self.positional = SinusoidalPositionalEncoding(config.positional_dropout_prob)
    else:
      self.positional = None

  def forward(
      self,
      examples: torch.Tensor,
      labels: torch.Tensor,
      *,
      is_training: bool,
  ) -> torch.Tensor:
    """Return token embeddings."""
    cfg = self.config
    if cfg.example_encoding == "embedding":
      example_tokens = self.example_encoder(examples.long())
    elif cfg.example_encoding == "linear":
      batch, seq_len = examples.shape[:2]
      example_tokens = self.example_encoder(
          examples.view(batch, seq_len, -1).to(torch.float32)
      )
    elif cfg.example_encoding == "resnet":
      batch, seq_len, height, width, channels = examples.shape
      if channels != cfg.input_channels:
        raise ValueError(
            f"Expected {cfg.input_channels} image channels, received {channels}."
        )
      reshaped = (
          examples.to(torch.float32)
          .permute(0, 1, 4, 2, 3)
          .reshape(batch * seq_len, channels, height, width)
      )
      self.example_encoder.train(is_training)
      features = self.example_encoder(reshaped)
      if self.resnet_projection is not None:
        features = self.resnet_projection(features)
      example_tokens = features.view(batch, seq_len, -1)
    else:
      raise ValueError(f"Unsupported example_encoding '{cfg.example_encoding}'")

    if cfg.example_dropout_prob > 0.0:
      example_tokens = F.dropout(
          example_tokens,
          p=cfg.example_dropout_prob,
          training=True,  # Always apply example noise, matching JAX behaviour.
      )

    labels_to_embed = labels
    if cfg.concatenate_labels:
      labels_to_embed = labels.clone()
      labels_to_embed[:, -1] = cfg.num_classes
    label_tokens = self.label_embedding(labels_to_embed.long())

    if cfg.concatenate_labels:
      tokens = torch.cat([example_tokens, label_tokens], dim=-1)
    else:
      batch, seq_len, dim = example_tokens.shape
      tokens = torch.zeros(
          batch,
          seq_len * 2 - 1,
          dim,
          dtype=example_tokens.dtype,
          device=example_tokens.device,
      )
      tokens[:, 0::2, :] = example_tokens
      tokens[:, 1::2, :] = label_tokens[:, :-1, :]

    if self.positional is not None:
      tokens = tokens + self.positional(tokens)

    return tokens


def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
  mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)
  return mask


class TransformerBlock(nn.Module):
  """Basic pre-norm transformer block with causal self attention."""

  def __init__(
      self,
      embed_dim: int,
      config: TransformerConfig,
  ):
    super().__init__()
    self.config = config
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(
        embed_dim,
        config.num_heads,
        dropout=config.dropout_prob,
        batch_first=True,
    )
    self.dropout = nn.Dropout(config.dropout_prob)
    widening_dim = embed_dim * config.widening_factor
    self.norm2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(
        nn.Linear(embed_dim, widening_dim),
        nn.GELU(),
        nn.Dropout(config.dropout_prob),
        nn.Linear(widening_dim, embed_dim),
        nn.Dropout(config.dropout_prob),
    )

  def forward(
      self,
      x: torch.Tensor,
      *,
      attn_mask: torch.Tensor,
      key_padding_mask: Optional[torch.Tensor],
  ) -> torch.Tensor:
    qkv = self.norm1(x)
    attn_out, _ = self.attn(
        qkv,
        qkv,
        qkv,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    x = x + self.dropout(attn_out)
    x = x + self.ffn(self.norm2(x))
    return x


class TransformerICL(nn.Module):
  """Full transformer model emitting class logits."""

  def __init__(
      self,
      embed_config: EmbeddingConfig,
      transformer_config: TransformerConfig,
      *,
      num_classes: int,
  ):
    super().__init__()
    self.embed_config = embed_config
    self.transformer_config = transformer_config
    self.embedder = InputEmbedder(embed_config)
    token_dim = embed_config.emb_dim * (2 if embed_config.concatenate_labels else 1)
    self.blocks = nn.ModuleList(
        TransformerBlock(token_dim, transformer_config)
        for _ in range(transformer_config.num_layers)
    )
    self.norm = nn.LayerNorm(token_dim)
    self.readout = nn.Linear(token_dim, num_classes)

  def forward(
      self,
      examples: torch.Tensor,
      labels: torch.Tensor,
      *,
      attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    tokens = self.embedder(examples, labels, is_training=self.training)
    seq_len = tokens.shape[1]
    attn_mask = _build_causal_mask(seq_len, tokens.device)
    padding_mask = None
    if attention_mask is not None:
      padding_mask = ~attention_mask.bool()
      if not self.embed_config.concatenate_labels:
        interleaved = torch.ones_like(tokens[..., 0], dtype=attention_mask.dtype)
        interleaved[:, 0::2] = attention_mask
        interleaved[:, 1::2] = attention_mask[:, :-1]
        padding_mask = ~interleaved.bool()

    for block in self.blocks:
      tokens = block(tokens, attn_mask=attn_mask, key_padding_mask=padding_mask)

    tokens = self.norm(tokens)
    logits = self.readout(tokens)
    return logits


__all__ = [
    "EmbeddingConfig",
    "TransformerConfig",
    "TransformerICL",
]
