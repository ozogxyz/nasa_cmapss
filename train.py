import hydra
import pytorch_lightning as pl
import rul_datasets
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import *

from src.models.regressor import ConvLSTMNet
from src.utils.callbacks import PrintCallback

@hydra.main(version_base=None, config_path='config/', config_name='config')
def main(cfg: DictConfig) -> None:
    print(cfg)
    pl.seed_everything(cfg.seed)

    # Load dataset
    dataset = rul_datasets.CmapssReader(cfg.dataset.fd)
    dm = rul_datasets.RulDataModule(dataset, batch_size=cfg.dataset.batch_size)

    # Build model
    model = ConvLSTMNet(cfg.model)
    model = model.to(cfg.device)

    # Train
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm=cfg.training.gradient_clip_algorithm,
        callbacks=[
            PrintCallback(cfg.print),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.ModelCheckpoint(*cfg.checkpoint)
        ]
    )

    trainer.fit(model, dm)

    # Save model
    torch.save(model.state_dict(), cfg.model.model_path)
    trainer.logger.log_hyperparams(cfg.training)
    print(f"Model saved to {cfg.model.model_path}")
    trainer.save_checkpoint()
    trainer.logger.close()

    # Load best model
    trainer.load_checkpoint()
    print(f"Best model loaded from {cfg.model.model_path}")
    trainer.test(model, dm)
    print(f"Test completed")

    # Save model
    print(f'Config saved to {cfg.config_path}')
    print(f'Log dir saved to {cfg.log_dir}')
    print(f'Version saved to {cfg.version}')
    print(f'Git sha saved to {cfg.git_sha}')
    print(f'Git branch saved to {cfg.git_branch}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'PyTorch available: {torch.__version__}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch version: {torch.version.cuda}')

    # Save config
    with open(cfg.config_path, 'w') as f:
        cfg.dump(cfg)
        f.write('\n')
        f.write(str(cfg))
        f.write('\n')
        f.write(str(model))
        f.write('\n')
        f.write(str(trainer))
        f.write('\n')
        f.write(str(cfg.training))
        f.write('\n')
        f.write(str(cfg.model))
        f.write('\n')
        f.write(str(cfg.log_dir))
        f.write('\n')
        f.write(str(cfg.version))
        f.write('\n')
        f.write(str(cfg.git_sha))
        f.write('\n')
        f.write(str(cfg.git_branch))

    print('Done!')


if __name__ == '__main__':
    main()
