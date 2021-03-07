import sys
import numpy as np
import torch
import torch.nn as nn
# TODO: make not hacky
sys.path.insert(0, "..")
from base import *

def train(model: nn.Module,
          loader: torch.utils.data.DataLoader,
          logger: logger.Logger,
          loss_fn,
          n_epochs: int=5,
          optim_kwargs: dict={},
          autoencoder: bool=True,):
    optim = torch.optim.Adam(model.parameters(), **optim_kwargs)
    losses = {}
    step_num = 0
    for _ in range(n_epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            if autoencoder:
                loss = model.loss(x_batch)
            else:
                loss = model.loss(y_batch)
            loss.backward()
            optimizer.step()
            losses[step_num] = loss.item()
            step_num += x_batch.shape[0]
    return 
            


if __name__=='__main__':
    task = data_loader.ImageNetDenoising()
    loader = task.loader(train=True)
    log = logger.Logger
    model = models.ConvDenoisingAutoencoder()

