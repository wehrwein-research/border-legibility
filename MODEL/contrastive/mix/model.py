from pathlib import Path

import torch
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import sys

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('border-legibility')+1])
sys.path.append(str(PROJ_ROOT) + '/MODEL/contrastive/mix')

from dataset import get_train_and_val_loader

sys.path.append(str(PROJ_ROOT) + '/MODEL/contrastive/siamese_experiments/')
from modelComparisonsSiamese import run_mix_model

'''
Main model used in train.py
'''
class SiameseDecider(pl.LightningModule):
    def __init__(self, img_file=None, n=None, split=None, lr=None, 
                 batch_size=4, weight_decay=1e-4):
        super().__init__()
        ''' The actual torch model (the SiameseDecider class is a pytorch lightning wrapper to make training simple)'''
        self.model = MLPHeadSiameseNet50()
        self.criterion = nn.CrossEntropyLoss() 
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.img_file = img_file
        self.n = n
        self.split = split
        self.batch_size = batch_size
        
        self.optim = None
        self.sched = None

        self.trainloader = None
        self.valloader = None
        self.get_loaders()

    def get_loaders(self, change_val=True):
        if self.img_file is not None:
            t, v = get_train_and_val_loader(self.img_file, self.n, 
                                            self.split, batch_size=self.batch_size,
                                            test=False)
            self.trainloader = t
            if change_val:
                self.valloader = v


    def forward(self, x1, x2):
        return self.model.forward(x1, x2)


    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr, 
                                weight_decay=self.weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=0.1,
            patience=4,
            threshold=1e-3,
            min_lr = 1e-7
        )
        #return {"optimizer": self.optim, "lr_scheduler": self.sched, "monitor": "train_loss"}
        return self.optim

        
    def train_dataloader(self):
        return self.trainloader 

    def val_dataloader(self):
        return self.valloader 
    
    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    ''' Use this to run modelComparisons at the end of every epoch and log the result.
        Don't do this with the any final test set. Only older test sets like the turk pilot.
    '''
#    def training_epoch_end(self, output):
#        acc = run_mix_model(id=self.current_epoch, model=self)
#        self.log('test_loss', 1-acc, on_epoch=True, prog_bar=False)

    
    
class MLPHeadSiameseNet(nn.Module):
    def __init__(self):
        super(MLPHeadSiameseNet, self).__init__() 
        main = models.resnet18(pretrained=False)
        main.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)

        self.back = nn.Sequential(*list(main.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, x1, x2):
        z1 = self.back(x1)
        z2 = self.back(x2)
        # concat features into each other for one batch with double the features 
        z = rearrange([z1, z2], 'n b c h w -> b (n c h w)')
        return self.head(z)

class MLPHeadSiameseNet50(nn.Module):
    def __init__(self):
        super(MLPHeadSiameseNet50, self).__init__() 
        main = models.resnet50(pretrained=True)
        main.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)


        self.back = nn.Sequential(*list(main.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2)
        )
        self.criterion = nn.CrossEntropyLoss() 
        
        
    def forward(self, x1, x2):
        z1 = self.back(x1)
        z2 = self.back(x2)
        # concat features into each other for one batch with double the features 
        z = rearrange([z1, z2], 'n b c h w -> b (n c h w)')
        return self.head(z)
    
class ConvHeadSiameseNet(nn.Module):
    def __init__(self):
        super(ConvHeadSiameseNet, self).__init__() 
        main = models.resnet50(pretrained=True)
        for name, param in main.named_parameters():
            param.requires_grad = False

        self.back = nn.Sequential(
            main.conv1,
            main.bn1,
            main.relu,
            main.maxpool
        )
        self.head = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=29, stride=1, bias=False),
            main.relu,
            nn.Conv3d(2, 2, kernel_size=27, stride=2, bias=False),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            Rearrange('b n c h w -> b (n c h w)')
        )
        self.criterion = nn.CrossEntropyLoss() 
        

    def forward(self, x1, x2):
        z1 = self.back(x1)
        z2 = self.back(x2)
        # concat features into each other for one batch with double the features 
        z = rearrange([z1, z2], 'n b c h w -> b n c h w')
        return self.head(z)
    
class SiameseDeciderOld(pl.LightningModule):
    def __init__(self, img_file=None, n=None, split=None, lr=None, 
                 batch_size=4, weight_decay=0):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.back = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.img_file = img_file
        self.n = n
        self.split = split
        if img_file is not None:
            t, v = get_train_and_val_loader(img_file, n, split, batch_size=batch_size)
        else:
            t, v = None, None
        self.trainloader = t
        self.valloader = v

    def forward(self, x1, x2):
        z1 = self.back(x1)
        z2 = self.back(x2)
        # concat features into each other for one batch with double the features 
        z = rearrange([z1, z2], 'n b c h w -> b (n c h w)')
        return self.head(z)


    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False) 
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def train_dataloader(self):
        return self.trainloader 

    def val_dataloader(self):
        return self.valloader 
    
    
    def training_epoch_end(self, output):
        acc = run_mix_model(id=self.current_epoch, model=self)
        self.log('test_loss', 1-acc)

