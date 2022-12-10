import torch
import rul_datasets

from models.cnn1d_lstm import Cnn1dLSTM
torch.manual_seed(42)


if __name__ == '__main__':
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
    dm.prepare_data()  # (1)!
    dm.setup()  # (2)!

    model_1 = Cnn1dLSTM()  # (3)!
    optim = torch.optim.Adam(model_1.parameters())

    best_val_loss = torch.inf

    for epoch in range(100):
        print(f"Train epoch {epoch}")
        model_1.train()
        for features, targets in dm.train_dataloader():
            optim.zero_grad()

            predictions = model_1(features)
            loss = torch.sqrt(torch.mean((targets - predictions)**2))  # (4)!
            loss.backward()
            print(f"Training loss: {loss}")

            optim.step()

        print(f"Validate epoch {epoch}")
        model_1.eval()
        val_loss = 0
        num_samples = 0
        for features, targets in dm.val_dataloader():
            predictions = model_1(features)
            loss = torch.sum((targets - predictions)**2)
            val_loss += loss.detach()
            num_samples += predictions.shape[0]
        val_loss = torch.sqrt(val_loss / num_samples)  # (5)!

        # if best_val_loss < val_loss:
        #     break
        # else:
        #     best_val_loss = val_loss
        #     print(f"Validation loss: {best_val_loss}")

    test_loss = 0
    num_samples = 0
    for features, targets in dm.test_dataloader():
        predictions = model_1(features)
        loss = torch.sqrt(torch.dist(predictions, targets))
        test_loss += loss.detach()
        num_samples += predictions.shape[0]
    test_loss = torch.sqrt(test_loss / num_samples)  # (6)!

    print(f"Test loss: {test_loss}")