import torch
from torch.utils.data import Dataset

from utils.loader import load_FD001


class CMAPSS_001(Dataset):
    def __init__(self, path=None, train=False, transform=None):
        self.path = path
        self.data, self.labels, _ = load_FD001()
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
