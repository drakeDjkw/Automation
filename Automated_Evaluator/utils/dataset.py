"""Simple dataset loader used for local testing.

Provides `RainfallDataset` compatible with the existing Evaluator/DataLoader usage.
If the provided path exists and is a .npy file, it will be loaded; otherwise a small
synthetic dataset is generated so evaluations can run end-to-end locally.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class RainfallDataset(Dataset):
    def __init__(self, path: str = "data/rainfall.npy", seq_len=5, num_samples=20, channels=1, height=32, width=32):
        self.path = path

        if os.path.exists(path) and path.endswith(".npy"):
            arr = np.load(path)
            # Expecting shape (N, seq_len, C, H, W) or similar; try to coerce
            if arr.ndim == 5:
                self.data = arr
            else:
                # reshape into (N, seq_len, C, H, W) conservatively
                n = max(1, arr.size // (seq_len * channels * height * width))
                self.data = arr.reshape((n, seq_len, channels, height, width))
        else:
            # Generate a small synthetic dataset
            self.data = np.random.rand(num_samples, seq_len, channels, height, width).astype(np.float32)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        # Return (input_seq, target) as tensors. For simplicity, make target the next frame (last frame)
        seq = self.data[idx]
        x = seq[:-1]  # all but last as input
        y = seq[-1]   # last frame as target

        # Convert to torch tensors
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        return x_t, y_t
