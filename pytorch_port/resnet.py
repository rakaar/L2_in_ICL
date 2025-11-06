"""Minimal ResNet implementation matching the Haiku SimpleResNet configuration."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlockV1(nn.Module):
    """ResNet V1 block with optional bottleneck and projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        use_projection: bool,
        bottleneck: bool,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.use_projection = use_projection
        self.bottleneck = bottleneck
        self.projection: nn.Module | None = None
        if use_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
            )

        channel_div = 4 if bottleneck else 1
        mid_channels = out_channels // channel_div

        conv0_stride = 1 if bottleneck else stride
        conv0_kernel = 1 if bottleneck else 3
        conv0_padding = 0 if bottleneck else 1

        layers = []
        layers.append(
            (
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=conv0_kernel,
                    stride=conv0_stride,
                    padding=conv0_padding,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_channels, eps=eps, momentum=momentum),
            )
        )
        layers.append(
            (
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=(stride if bottleneck else 1),
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_channels, eps=eps, momentum=momentum),
            )
        )

        if bottleneck:
            conv2 = nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            bn2 = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
            nn.init.zeros_(bn2.weight)
            layers.append((conv2, bn2))

        self.layers = nn.ModuleList(
            nn.Sequential(conv, bn) for conv, bn in layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.use_projection and self.projection is not None:
            identity = self.projection(x)

        out = x
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx < num_layers - 1:
                out = F.relu(out, inplace=True)

        out = out + identity
        return F.relu(out, inplace=True)


class ResidualBlockGroup(nn.Module):
    """Stack of residual blocks with shared output width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int,
        stride: int,
        use_projection: bool,
        bottleneck: bool,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        blocks = []
        for block_idx in range(num_blocks):
            block_in = in_channels if block_idx == 0 else out_channels
            block_stride = stride if block_idx == 0 else 1
            blocks.append(
                ResidualBlockV1(
                    block_in,
                    out_channels,
                    stride=block_stride,
                    use_projection=(block_idx == 0 and use_projection),
                    bottleneck=bottleneck,
                    momentum=momentum,
                    eps=eps,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class SimpleResNet(nn.Module):
    """PyTorch port of the Haiku SimpleResNet used in the experiment."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        blocks_per_group: Sequence[int] = (2, 2, 2, 2),
        channels_per_group: Sequence[int] = (64, 128, 256, 512),
        strides: Sequence[int] = (1, 2, 2, 2),
        use_projection: Sequence[bool] = (True, True, True, True),
        bottleneck: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        flatten_superpixels: bool = False,
    ):
        super().__init__()
        if not (
            len(blocks_per_group)
            == len(channels_per_group)
            == len(strides)
            == len(use_projection)
        ):
            raise ValueError("ResNet configuration lists must all be length 4.")

        self.flatten_superpixels = flatten_superpixels

        self.initial_conv = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.initial_bn = nn.BatchNorm2d(64, eps=eps, momentum=momentum)

        in_channels = 64
        groups = []
        for blocks, channels, stride, proj in zip(
            blocks_per_group, channels_per_group, strides, use_projection
        ):
            groups.append(
                ResidualBlockGroup(
                    in_channels,
                    channels,
                    num_blocks=blocks,
                    stride=stride,
                    use_projection=proj,
                    bottleneck=bottleneck,
                    momentum=momentum,
                    eps=eps,
                )
            )
            in_channels = channels

        self.block_groups = nn.ModuleList(groups)

        self.final_bn: nn.Module | None = None
        if False:  # Placeholder for potential ResNet v2 support.
            self.final_bn = nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum)

        self.output_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_conv(x)
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
