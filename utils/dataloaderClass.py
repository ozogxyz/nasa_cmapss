import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessData import load


def subset_ind(dataset, ratio: float):
    return np.random.choice(len(dataset), size=int(ratio * len(dataset)), replace=False)


class CMAPSS(Dataset):
    def __init__(self, path=None, train=False, transform=None):
        self.data, self.labels, _, _ = load(path)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        label = self.labels[index]
        if self.transform is not None:
            x = self.transform(x)
        return torch.tensor(x), torch.tensor(label)
