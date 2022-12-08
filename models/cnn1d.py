import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class Supervised(LightningModule):
    def __init__(self):
        super(Supervised, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(nn.Linear(2, 1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.05)

    def forward(self, features):
        return self.regressor(self.feature_extractor(features))

    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds / 50, labels / 50)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx = None):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, dataloader_idx = None):
        features, labels = batch
        preds = self.forward(features)
        loss = nn.functional.mse_loss(preds, labels)
        self.log(f"test_loss_fd{dataloader_idx+1}", loss)
