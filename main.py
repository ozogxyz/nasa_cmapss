import argparse
import rul_datasets
import warnings
import pytorch_lightning as pl
from models.cnn1d_lstm import Cnn1dLSTM
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(42)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

if __name__ == "__main__":
    # read command line arguments
    parser = argparse.ArgumentParser(
        prog="Trainer",
        description="Runs the trainer for selected models.",
        epilog="See individual options with --help."
    )
    parser.add_argument('-fd', '--filename', help='[Data subset number of NASA CMAPSS Turbofan Engine data. '
                                                  'Default is subset 1.]', default=1, type=int)
    parser.add_argument('-b', '--batch_size', help='[Batch size to send to the trainer. '
                                                   'Default is 200.]', default=200, type=int)
    parser.add_argument('-e', '--epochs', help='[Maximum number of iterations for the trainer. '
                                               'Default is 10.]', default=10, type=int)
    parser.add_argument('--flatten', help='[Flatten the output of Conv1d layer output.]', default=False, type=bool)

    args = vars(parser.parse_args())

    fd = args['filename']
    batch_size = args['batch_size']
    max_epochs = args['epochs']

    # read data
    cmapss_fd1 = rul_datasets.CmapssReader(fd)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=batch_size)
    # create model
    model = Cnn1dLSTM(        
        batch_size = batch_size,
        in_channels = 14,
        out_channels = 64,
        kernel_size = 5,
        maxpool_kernel = 3,
        num_classes = 1,
        input_size = 14,
        hidden_size = 96,
        num_layers = 2,
        maxpool_stride = 2,
        window_size = 30,
        flatten = False,)
    # create trainer context
    logger = TensorBoardLogger("tb_logs", name="Cnn1dLSTM")
    trainer = pl.Trainer(accelerator='auto',max_epochs=max_epochs, logger=logger)
    # fit & test
    trainer.fit(model, dm)
    trainer.test(model, dm)
