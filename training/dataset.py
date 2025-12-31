import numpy as np
import torch

class BinDataset(torch.utils.data.Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.data[idx : idx + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64)
        )
        return x, y
