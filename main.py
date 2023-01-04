import logging

import hydra
import hydra_zen
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, EarlyStopping, Timer, ModelCheckpoint
from rul_datasets.core import RulDataModule
from rul_datasets.reader.cmapss import CmapssReader

from src.core.network import Network

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # load dataset
    reader = CmapssReader(cfg.dataset.filename)
    dm = RulDataModule(reader=reader, batch_size=cfg.dataset.batch_size)
    
    logger.info(f"Loaded data from {cfg.dataset.filename}")
    
    # train model
    model = Network(**cfg.model)

    # callbacks
    early_stopping = EarlyStopping(
        check_on_train_epoch_end=True,
        monitor="val_loss",
        patience=cfg.training.patience,
        strict=False,
        verbose=True,
        min_delta=cfg.training.min_delta,
        mode='min'
    )
    ckpt = ModelCheckpoint(cfg.training.save_dir)
    summary = RichModelSummary()
    progress = RichProgressBar()
    timer = Timer()

    callbacks = [ckpt, early_stopping, progress, timer, summary]

    trainer = Trainer(
        accelerator='mps', 
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm=cfg.training.gradient_clip_algorithm,
        max_steps=cfg.training.max_steps,
        default_root_dir=cfg.training.lightning_log_dir
    )
    trainer.fit(model, dm)

    # test
    trainer.test(model, dm, ckpt_path="best")


if __name__ == "__main__":
    main()
