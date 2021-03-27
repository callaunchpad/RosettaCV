"""
General utils for the project
"""
import torch

import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from typing import Callable


def get_loss_on_dataloader(model: nn.Module, data_loader: DataLoader, loss_fn: Callable) -> torch.Tensor:
    """
    Gets the average loss over all batches on the given dataloader
    :param model: The model to predict with
    :param data_loader: The dataloader to evaluate over
    :param loss_fn: The loss function to apply to the predictions
    :return:
    """
    # Find the device the model is on
    device = next(model.parameters()).device
    total_loss = 0

    for batch in data_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        predictions = model(inputs)

        total_loss += loss_fn(predictions, labels)

    return total_loss / len(data_loader)

