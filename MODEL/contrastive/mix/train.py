import sys, json
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset import BordersDataset, get_train_and_val_loader
from model import SiameseDecider, SiameseDeciderOld
sys.path.append('../../../data')

PROJ_ROOT = str(Path(*Path.cwd().parts[:Path().cwd().parts.index('border-legibility')+1]))
SAVE_DIR = PROJ_ROOT + '/MODEL/contrastive/mix/weights/'

def get_config(args):
    cfg = args[1]
    cfg = json.load(open(cfg))
    return cfg

def main():
    we_logging = True if len(sys.argv) < 3 else sys.argv[2] != 'no'
    print('We logging?:', we_logging)
    num = '' if len(sys.argv) < 4 else sys.argv[3] 
    cfg = get_config(sys.argv)
    
    logger = pl.loggers.wandb.WandbLogger(name=Path(cfg["name"]).stem+num, 
                                          project="mix-models") if we_logging else False
    checkpoint = ModelCheckpoint(
        dirpath=SAVE_DIR,
        monitor='test_loss',
        filename=Path(cfg["name"]).stem + num
    )

    model = SiameseDeciderOld(cfg['img_list'], cfg['n'], cfg['ratio'],
                           cfg['lr'], cfg['batch_size'], cfg['weight_decay'])
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        accelerator = "cpu",
        max_epochs=cfg["num_epochs"],
        callbacks=[checkpoint],
        logger=logger,
        num_sanity_val_steps=2
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
