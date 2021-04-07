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

        total_loss += loss_fn(predictions, labels).item()

    return total_loss / len(data_loader)

def get_accuracy_on_dataloader(model: nn.Module, data_loader: DataLoader) -> torch.Tensor:
    """
    Gets the accuracy over all batches on the given dataloader
    :param model: The model to predict with
    :param data_loader: The dataloader to evaluate over
    :return: The accuracy of the model on a scale from 0 to 1.
    """
    # Find the device the model is on
    device = next(model.parameters()).device
    total_items = 0
    total_correct = 0

    for batch in data_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        predictions = model(inputs)
        _, idxs = torch.max(predictions, dim=1)
        total_items += len(idxs)
        for i in range(len(idxs)):
            if idxs[i] == labels[i]:
                total_correct += 1

    return total_correct / total_items
