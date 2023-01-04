import logging

import hydra
import hydra_zen
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
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

    trainer = Trainer(
        accelerator='mps', 
        max_epochs=cfg.training.max_epochs,
        callbacks=[RichModelSummary(), RichProgressBar()],
    )
    trainer.fit(model, dm)

    # evaluate
    evaluator = hydra_zen.Evaluator(model, dm)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
