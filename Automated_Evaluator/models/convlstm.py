"""Minimal mock ConvLSTM model for local testing.

Defines `ConvLSTM` which is a tiny PyTorch nn.Module that accepts an input tensor
and returns an output tensor shaped like the expected target. Also exposes an alias
`Convlstm` to match the evaluation inference naming heuristics.
"""

import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden=8):
        super().__init__()
        # A very small conv net to allow forward pass during tests
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Expect x shape: (B, T, C, H, W) or (T, C, H, W) for single sample
        if x.dim() == 5:
            # apply to last time step or average across time
            last = x[:, -1]
        elif x.dim() == 4:
            last = x
        else:
            # try to coerce
            last = x.view(-1, 1, int(x.size(-2)), int(x.size(-1)))

        return self.net(last)


# alias matching the CamelCase heuristic used by evaluate.run_evaluation
Convlstm = ConvLSTM
