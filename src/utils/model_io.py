"""
IO operations for models
"""
import torch
import utils.data_io as data

import utils.data_io as io

from torch.nn import Module
from typing import Tuple
# from torch.utils.tensorboard import SummaryWriter
from numpy import ndarray


def save_model_checkpoint(path: str, model: Module, optimizer=None) -> None:
    """
    Saves a model and optionally an optimizer for checkpointing
    :param path: The path to save to, rooted at /saved
    :param model: The model to save
    :param optimizer: The optimizer state to save
    :return: None
    """
    save_dict = {'model': model.state_dict()}

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    torch.save(save_dict, f"{data.get_model_path()}/{path}")


def load_model_checkpoint(path: str) -> Tuple:
    """
    Loads the model and optimizer save dicts if possible
    :param path: The path to load from
    :return: A list of the form [model_save_dict, optimizer_save_dict]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"{data.get_model_path()}/{path}", map_location=device)

    return checkpoint['model'], checkpoint.get('optimizer', None)


def visualize_model(model: Module, images: ndarray, experiment_name: str = "default") -> None:
    """
    Visualizes the given model in tensorboard
    :param model: The model to visualize
    :return: None
    """
    raise NotImplementedError()

    writer = SummaryWriter(f"{io.get_model_path()}/tensorboard/{experiment_name}")
    writer.add_graph(model, images)
    writer.close()

