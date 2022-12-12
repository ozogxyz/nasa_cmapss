import torch
import rul_datasets
import yaml
from models.cnn1d_lstm import Cnn1dLSTM
torch.manual_seed(42)

cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
dm.prepare_data()  # (1)!
dm.setup()  # (2)!

if __name__ == '__main__':
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    # dataset params
    fd = params.get('dataset').get('filename')
    batch_size = params.get('dataset').get('batch_size')

    # model params
    in_channels = params.get('model').get('in_channels')
    out_channels = params.get('model').get('out_channels')
    kernel_size = params.get('model').get('kernel_size')
    maxpool_kernel = params.get('model').get('maxpool_kernel')
    num_classes = params.get('model').get('num_classes')
    hidden_size = params.get('model').get('hidden_size')
    num_layers = params.get('model').get('num_layers')
    maxpool_stride = params.get('model').get('maxpool_stride')
    window_size = params.get('model').get('window_size')

    # training params
    lr = params.get('training').get('lr')
    max_epochs = params.get('training').get('epochs')
    patience = params.get('training').get('patience')
    min_delta = params.get('training').get('min_delta')

    ####################################
    # 1. Create Model
    ####################################
    model = Cnn1dLSTM(
        batch_size=batch_size,
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
    )

    optim = torch.optim.Adam(model.parameters())

    best_val_loss = torch.inf

    for epoch in range(100):
        print(f"Train epoch {epoch}")
        model.train()
        for features, targets in dm.train_dataloader():
            optim.zero_grad()

            predictions = model(features)
            loss = torch.sqrt(torch.mean((targets - predictions)**2))  # (4)!
            loss.backward()
            print(f"Training loss: {loss}")

            optim.step()

        print(f"Validate epoch {epoch}")
        model.eval()
        val_loss = 0
        num_samples = 0
        for features, targets in dm.val_dataloader():
            predictions = model(features)
            loss = torch.sum((targets - predictions)**2)
            val_loss += loss.detach()
            num_samples += predictions.shape[0]
        val_loss = torch.sqrt(val_loss / num_samples)  # (5)!

        if best_val_loss < val_loss:
            break
        else:
            best_val_loss = val_loss
            print(f"Validation loss: {best_val_loss}")

    test_loss = 0
    num_samples = 0
    for features, targets in dm.test_dataloader():
        predictions = model(features)
        loss = torch.sqrt(torch.dist(predictions, targets))
        test_loss += loss.detach()
        num_samples += predictions.shape[0]
    test_loss = torch.sqrt(test_loss / num_samples)  # (6)!

    print(f"Test loss: {test_loss}")