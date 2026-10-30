"""PyTorch modules for the DNN and the Kolmogorov-Arnold Network (KAN).

What this file does
-------------------
Defines the network architectures only (no training code; see neural.py):
``build_mlp`` makes a ReLU/dropout feed-forward network ending in one logit, ``KANLayer`` is one
Kolmogorov-Arnold layer in which every input-output edge is a learnable univariate function
(a SiLU base term plus a cubic B-spline on a fixed grid), and ``build_kan`` stacks such layers.
Because each edge is a one-dimensional curve it can be plotted, which is why a KAN sits between
logistic regression and a black-box DNN in the interpretability trade-off.

Imports torch at module level, so import this file lazily (``neural.py`` does).
The KAN layer follows the efficient-KAN formulation: each edge carries a learnable
univariate function = SiLU base term + B-spline term on a fixed grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                 spline_order: int = 3, grid_range: tuple = (-3.0, 3.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(in_features))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion. x: (B, in) -> (B, in, grid_size + spline_order)."""
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.spline_weight.reshape(self.out_features, -1),
        )
        return base + spline


def build_mlp(in_dim: int, hidden=(128, 64, 32), dropout=(0.30, 0.20, 0.10)) -> nn.Module:
    layers, prev = [], in_dim
    for units, p in zip(hidden, dropout):
        layers += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(p)]
        prev = units
    layers.append(nn.Linear(prev, 1))  # logit
    return nn.Sequential(*layers)


def build_kan(in_dim: int, hidden=(8,), grid_size: int = 5, spline_order: int = 3) -> nn.Module:
    dims = [in_dim, *hidden, 1]
    return nn.Sequential(*[
        KANLayer(a, b, grid_size=grid_size, spline_order=spline_order)
        for a, b in zip(dims[:-1], dims[1:])
    ])  # logit
