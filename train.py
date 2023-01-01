import hydra
import pytorch_lightning as pl
import rul_datasets
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import *

from src.rul_models import rul_models


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:

    # Instantiate config
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
        callbacks=[RichModelSummary(), RichProgressBar()],
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
    )

    trainer.fit(model, dm)
    # trainer.test(model, dm)


if __name__ == '__main__':
    main()
    print('done')
