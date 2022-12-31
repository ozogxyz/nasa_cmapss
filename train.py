import hydra
import pytorch_lightning as pl
import rul_datasets
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import *
from hydra.utils import instantiate
from src.rul_models import rul_models


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:

    cfg = instantiate(cfg)

    # Load dataset
    data = rul_datasets.CmapssReader(cfg.data.fd)
    dm = rul_datasets.RulDataModule(data, batch_size=cfg.data.batch_size)

    # Load model
    model = rul_models.RulModel(cfg.model)

    # Train
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    main()
