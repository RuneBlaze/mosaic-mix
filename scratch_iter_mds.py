"""Scratch script."""

from streaming import StreamingDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = DataLoader(StreamingDataset(local="data"), batch_size=1, num_workers=2)

    for record in dataset:
        print(record)
