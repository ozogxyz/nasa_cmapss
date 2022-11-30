import pytorch_lightning as pl
import rul_datasets

from models import lstm as rul_estimator

# import rul_estimator
# for fd in [1, 2, 3, 4]:

for fd in[1]:
    cmapss = rul_datasets.CmapssReader(fd)
    dm = rul_datasets.RulDataModule(cmapss, batch_size=32)
    
    my_rul_estimator = rul_estimator.LSTM(16, 96, 4)
    
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=100,
        )
    trainer.fit(my_rul_estimator, dm)
    
    trainer.test(my_rul_estimator, dm)
