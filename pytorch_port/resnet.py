"""Minimal ResNet implementation that mirrors Haiku's SimpleResNet."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _resolve_bn_config(bn_config: Mapping[str, Any] | None) -> Dict[str, Any]:
  config: Dict[str, Any] = dict(bn_config or {})
  config.setdefault("decay_rate", 0.9)
  config.setdefault("eps", 1e-5)
  config.setdefault("create_scale", True)
  config.setdefault("create_offset", True)
  return config


def _make_batchnorm(
    num_features: int,
    bn_config: Mapping[str, Any],
    *,
    zero_init: bool = False,
) -> nn.BatchNorm2d:
  decay = float(bn_config.get("decay_rate", 0.9))
  eps = float(bn_config.get("eps", 1e-5))
  create_scale = bool(bn_config.get("create_scale", True))
  create_offset = bool(bn_config.get("create_offset", True))
  affine = create_scale or create_offset

  bn = nn.BatchNorm2d(
      num_features,
      eps=eps,
      momentum=1.0 - decay,
      affine=affine,
  )
  if affine and not create_scale:
    nn.init.ones_(bn.weight)
    bn.weight.requires_grad = False
  if affine and not create_offset:
    nn.init.zeros_(bn.bias)
    bn.bias.requires_grad = False
  if affine and zero_init:
    nn.init.zeros_(bn.weight)
  return bn


def _init_conv(conv: nn.Conv2d) -> None:
  nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
  if conv.bias is not None:
    nn.init.zeros_(conv.bias)


def _same_padding(kernel_size: int) -> int:
  return (kernel_size - 1) // 2


class BlockV1(nn.Module):
  """ResNet v1 block with optional bottleneck/projection."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      *,
      stride: int,
      use_projection: bool,
      bottleneck: bool,
      bn_config: Mapping[str, Any],
  ):
    super().__init__()
    self.use_projection = use_projection

    channel_div = 4 if bottleneck else 1
    mid_channels = max(1, out_channels // channel_div)

    if self.use_projection:
      self.proj_conv = nn.Conv2d(
          in_channels,
          out_channels,
          kernel_size=1,
          stride=stride,
          bias=False,
      )
      _init_conv(self.proj_conv)
      self.proj_bn = _make_batchnorm(out_channels, bn_config)
    else:
      self.proj_conv = None
      self.proj_bn = None

    layers = []
    conv0 = nn.Conv2d(
        in_channels,
        mid_channels,
        kernel_size=(1 if bottleneck else 3),
        stride=(1 if bottleneck else stride),
        padding=(0 if bottleneck else _same_padding(3)),
        bias=False,
    )
    _init_conv(conv0)
    bn0 = _make_batchnorm(mid_channels, bn_config)
    layers.append(nn.Sequential(conv0, bn0))

    conv1 = nn.Conv2d(
        mid_channels,
        mid_channels,
        kernel_size=3,
        stride=(stride if bottleneck else 1),
        padding=_same_padding(3),
        bias=False,
    )
    _init_conv(conv1)
    bn1 = _make_batchnorm(mid_channels, bn_config)
    layers.append(nn.Sequential(conv1, bn1))

    if bottleneck:
      conv2 = nn.Conv2d(
          mid_channels,
          out_channels,
          kernel_size=1,
          stride=1,
          bias=False,
      )
      _init_conv(conv2)
      bn2 = _make_batchnorm(out_channels, bn_config, zero_init=True)
      layers.append(nn.Sequential(conv2, bn2))

    self.layers = nn.ModuleList(layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shortcut = x
    if self.use_projection and self.proj_conv is not None:
      shortcut = self.proj_conv(shortcut)
      shortcut = self.proj_bn(shortcut)

    out = x
    for idx, layer in enumerate(self.layers):
      out = layer(out)
      if idx < len(self.layers) - 1:
        out = F.relu(out, inplace=True)

    return F.relu(out + shortcut, inplace=True)


class BlockV2(nn.Module):
  """ResNet v2 block with optional bottleneck/projection."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      *,
      stride: int,
      use_projection: bool,
      bottleneck: bool,
      bn_config: Mapping[str, Any],
  ):
    super().__init__()
    self.use_projection = use_projection

    channel_div = 4 if bottleneck else 1
    mid_channels = max(1, out_channels // channel_div)

    if self.use_projection:
      self.proj_conv = nn.Conv2d(
          in_channels,
          out_channels,
          kernel_size=1,
          stride=stride,
          bias=False,
      )
      _init_conv(self.proj_conv)
    else:
      self.proj_conv = None

    self.layers = nn.ModuleList()

    conv0 = nn.Conv2d(
        in_channels,
        mid_channels,
        kernel_size=(1 if bottleneck else 3),
        stride=(1 if bottleneck else stride),
        padding=(0 if bottleneck else _same_padding(3)),
        bias=False,
    )
    _init_conv(conv0)
    bn0 = _make_batchnorm(in_channels, bn_config)
    self.layers.append(nn.ModuleList([bn0, conv0]))

    conv1 = nn.Conv2d(
        mid_channels,
        mid_channels,
        kernel_size=3,
        stride=(stride if bottleneck else 1),
        padding=_same_padding(3),
        bias=False,
    )
    _init_conv(conv1)
    bn1 = _make_batchnorm(mid_channels, bn_config)
    self.layers.append(nn.ModuleList([bn1, conv1]))

    if bottleneck:
      conv2 = nn.Conv2d(
          mid_channels,
          out_channels,
          kernel_size=1,
          stride=1,
          bias=False,
      )
      _init_conv(conv2)
      bn2 = _make_batchnorm(mid_channels, bn_config)
      self.layers.append(nn.ModuleList([bn2, conv2]))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shortcut = x
    out = x
    for idx, (bn, conv) in enumerate(self.layers):
      out = bn(out)
      out = F.relu(out, inplace=True)
      if idx == 0 and self.use_projection and self.proj_conv is not None:
        shortcut = self.proj_conv(out)
      out = conv(out)
    return out + shortcut


class BlockGroup(nn.Module):
  """Stack of residual blocks mirroring Haiku's BlockGroup."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      *,
      num_blocks: int,
      stride: int,
      bn_config: Mapping[str, Any],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
  ):
    super().__init__()
    block_cls = BlockV2 if resnet_v2 else BlockV1
    blocks = []
    for block_idx in range(num_blocks):
      block_in = in_channels if block_idx == 0 else out_channels
      block_stride = stride if block_idx == 0 else 1
      blocks.append(
          block_cls(
              block_in,
              out_channels,
              stride=block_stride,
              use_projection=(block_idx == 0 and use_projection),
              bottleneck=bottleneck,
              bn_config=bn_config,
          )
      )
    self.blocks = nn.ModuleList(blocks)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      x = block(x)
    return x


def _check_length(name: str, value: Sequence[Any], expected: int = 4) -> None:
  if len(value) != expected:
    raise ValueError(f"`{name}` must be of length {expected} not {len(value)}")


def _resolve_padding(padding: Any, kernel: int) -> int:
  if isinstance(padding, str):
    padding = padding.upper()
    if padding == "SAME":
      return _same_padding(kernel)
    if padding == "VALID":
      return 0
    raise ValueError(f"Unsupported padding mode '{padding}'.")
  return int(padding)


class SimpleResNet(nn.Module):
  """PyTorch port of Haiku's SimpleResNet."""

  def __init__(
      self,
      *,
      input_channels: int = 1,
      blocks_per_group: Sequence[int] = (2, 2, 2, 2),
      bn_config: Mapping[str, Any] | None = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] | None = None,
      initial_conv_config: Mapping[str, Any] | None = None,
      strides: Sequence[int] = (1, 2, 2, 2),
      flatten_superpixels: bool = False,
  ):
    super().__init__()

    bn_config = _resolve_bn_config(bn_config)
    use_projection = use_projection or (True, True, True, True)

    _check_length("blocks_per_group", blocks_per_group)
    _check_length("channels_per_group", channels_per_group)
    _check_length("strides", strides)
    _check_length("use_projection", use_projection)

    init_conv_cfg: MutableMapping[str, Any] = dict(initial_conv_config or {})
    init_conv_cfg.setdefault("output_channels", 64)
    init_conv_cfg.setdefault("kernel_shape", 7)
    init_conv_cfg.setdefault("stride", 2)
    init_conv_cfg.setdefault("with_bias", False)
    init_conv_cfg.setdefault("padding", "SAME")

    init_out = int(init_conv_cfg["output_channels"])
    kernel = int(init_conv_cfg["kernel_shape"])
    stride = int(init_conv_cfg["stride"])
    padding = _resolve_padding(init_conv_cfg.get("padding", "SAME"), kernel)
    bias = bool(init_conv_cfg.get("with_bias", False))

    self.initial_conv = nn.Conv2d(
        input_channels,
        init_out,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    _init_conv(self.initial_conv)

    self.resnet_v2 = resnet_v2
    if not resnet_v2:
      self.initial_bn = _make_batchnorm(init_out, bn_config)
    else:
      self.initial_bn = None

    in_channels = init_out
    groups = []
    for blocks, channels, stride_val, proj in zip(
        blocks_per_group, channels_per_group, strides, use_projection
    ):
      groups.append(
          BlockGroup(
              in_channels,
              channels,
              num_blocks=blocks,
              stride=stride_val,
              bn_config=bn_config,
              resnet_v2=resnet_v2,
              bottleneck=bottleneck,
              use_projection=proj,
          )
      )
      in_channels = channels

    self.block_groups = nn.ModuleList(groups)
    self.flatten_superpixels = flatten_superpixels
    self.final_bn = _make_batchnorm(in_channels, bn_config) if resnet_v2 else None
    self.output_channels = in_channels

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.initial_conv(x)
    if not self.resnet_v2:
      out = self.initial_bn(out)
      out = F.relu(out, inplace=True)

    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

    for group in self.block_groups:
      out = group(out)

    if self.final_bn is not None:
      out = self.final_bn(out)
      out = F.relu(out, inplace=True)

    if self.flatten_superpixels:
      out = torch.flatten(out, start_dim=1)
    else:
      out = out.mean(dim=(2, 3))
    return out

  @property
  def output_dim(self) -> int:
    if self.flatten_superpixels:
      raise AttributeError(
          "Output dimension is spatially dependent when flattening superpixels."
      )
    return self.output_channels
