import rul_datasets

from models.models import *

if __name__ == "__main__":
    reader = rul_datasets.CmapssReader(fd=1)
    dev_features, dev_targets = reader.load_split("dev")
    print(f'dev features: {dev_features[0].shape}')
    x = torch.tensor(dev_features[0][:32], dtype=torch.float)
    model = Network(in_channels=30, out_channels=32, kernel_size=5)
    out = model.forward(x)
