"""
Code to train the models
"""

import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import wandb
import typing

from torch.utils.data import DataLoader
from train.train_callbacks import register_save_best_validation_loss


def inner_train_loop(model, optimizer, data_loaders, loss_fn, device, num_iters):
    """Inner training loop for meta training procedure.

    Args:
        model (torch.nn.Module): model to train
        optimizer (torch.optim): optimizer
        data_loaders (dict): dictionary containing "train": train data loader 
        loss_fn (torch.function): loss function
        device (torch.device): gpu if enabled
        num_iters (int): [description]
    """
    iters = 0
    while iters < num_iters:
        for sample in data_loaders['train']:
            if iters < num_iters:
                break

            inputs = sample['input']
            labels = sample['output']

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                preds = model(inputs)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

            # statistics
            wandb.log({'inner_loss': loss})
            print("Loss ", loss)

            iters += 1

    return model


def outer_train_loop(model, optimizer, train_tasks, val_tasks, loss_fn,
                     device, num_iters_inner, num_iters_outer,
                     update_parameters, update_param_kwargs=None):
    """Meta train a model

    Args:
        model (nn.Module): Initialized model to meta-train.
        optimizer (torch.optim): Optimizer to train inner loop.
        train_tasks (arr): Array of Task objects. 
        val_tasks (arr): Array of Task objects for validation.
        batch_size (int): batch size for data loaders.
        loss_fn (arr): Array of loss functions for inner loop training.
        device (torch.device): Device to train on. 
        num_iters_inner (int): Number of training iterations in the inner loop. 
                               One step = one parameter update in the inner loop.
        num_iters_outer (int): Number of outer steps. How many times the init parameters are updated.
        update_parameters (function): Function to update the init params. This should take in fn(model, init_params, new_params)

    Returns:
        nn.parameter: Final initialization parameters.
    """
    since = time.time()

    init_params = model.state_dict()
    iters = 0

    while iters < num_iters_outer:
        delta_params = init_params
        for task in all_data_loaders:
            if iters > num_iters_outer:
                break

            # Inner train using each dataloader received
            model.load_state_dict(init_params)

            dataloaders = {"train": task.loader("train"), "val": task.loader("val")}
            loss_fn = task.loss_fn
            # TODO do i have to re-instantiate optimizer
            new_model = inner_train_loop(model, optimizer, 
                                         dataloaders, loss_fn, 
                                         device, num_iters_inner)
            
            new_params = new_model.state_dict()

            # Update initial parameters using update_parameters function
            delta_params += update_parameters(model, init_params, new_params, loss_fn, update_param_kwargs)

            iters += 1

        init_params = delta_params


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return init_params
