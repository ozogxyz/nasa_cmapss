import torch
import torchmetrics
import rul_datasets
import yaml
from models.cnn1d_lstm import Cnn1dLSTM
from models.network import Network
from utils.read_params import read_params

torch.manual_seed(42)

PARAMS_FILEPATH = 'params.yaml'

if __name__ == "__main__":
    ####################################
    # 0. Read Hyper-Parameters
    fd, batch_size, window_size, in_channels, out_channels, kernel_size, maxpool_kernel, maxpool_stride, num_classes, hidden_size, num_layers, max_epochs, patience, min_delta, lr, dropout = read_params(PARAMS_FILEPATH)
    
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
    dm.prepare_data()  # (1)!
    dm.setup()  # (2)!
    
    ####################################
    # 1. Create Model
    ####################################
    model = Network(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        maxpool_kernel=maxpool_kernel,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        maxpool_stride=maxpool_stride,
        window_size=window_size,
        lr=lr,
        dropout=dropout,
    )

    optim = torch.optim.Adam(model.parameters())
    metric = torchmetrics.MeanSquaredError(squared=False)
    best_val_loss = torch.inf

    for epoch in range(1000):
        print(f"Train epoch {epoch}")
        model.train()
        for features, targets in dm.train_dataloader():
            optim.zero_grad()

            predictions = model(features)
            # loss = torch.sqrt(torch.mean((targets - predictions)**2))  # (4)!
            loss = metric(predictions, targets)
            loss.backward()
            # print(f"Training loss: {loss:.3f}")

            optim.step()

        print(f"Validate epoch {epoch}")
        model.eval()
        val_loss = 0
        num_samples = 0
        for features, targets in dm.val_dataloader():
            predictions = model(features)
        #     loss = torch.sum((targets - predictions)**2)
        #     val_loss += loss.detach()
        #     num_samples += predictions.shape[0]
        # val_loss = torch.sqrt(val_loss / num_samples)  # (5)!
            val_loss = metric(predictions, targets)

        if best_val_loss < val_loss:
            break
        else:
            best_val_loss = val_loss
            print(f"Validation loss: {best_val_loss:.4f}")

    test_loss = 0
    num_samples = 0
    for features, targets in dm.test_dataloader():
        predictions = model(features)
    #     loss = torch.sqrt(torch.dist(predictions, targets))
    #     test_loss += loss.detach()
    #     num_samples += predictions.shape[0]
    # test_loss = torch.sqrt(test_loss / num_samples)  # (6)!
        test_loss = metric(predictions, targets)

    print(f"Test loss: {test_loss:.4f}")