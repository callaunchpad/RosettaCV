"""
Code to train the models
"""

import torch

import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import wandb
import typing
import time

def maml_inner_train(model, optimizer, data_loaders, loss_fn, device, num_iters):
    return None

def maml_update_parameters(model, init_params, loss):
    # TODO
    return

def maml_outer_train_loop(model, optimizer, train_tasks, 
                        validation_tasks, batch_size,
                        num_iters_inner, num_iters_outer,
                        device, save_to, update_param_kwargs=None):
    """Meta train a model using MAML
    This is separate from the other meta train loop since this is written in such a way to conserve memory

    Args:
        model (nn.Module): Initialized model to meta-train.
        optimizer (torch.optim): Optimizer to train inner loop.
        train_tasks (arr): Array of Task objects. 
        validation_tasks (arr): Array of Task objects for validation.
        batch_size (int): batch size for data loaders.
        num_iters_inner (int): Number of training iterations in the inner loop. 
                               One step = one parameter update in the inner loop.
        num_iters_outer (int): Number of outer steps. How many times the init parameters are updated.
        device (torch.device): Device to train on. 
        save_to (str): Name of save file. Will save model with best validation accuracy.

    Returns:
        nn.parameter: Final initialization parameters.
    """
    since = time.time()

    init_params = model.state_dict()
    iters = 0
    min_val_loss = 10000

    all_data_loaders = []
    loss_fns = []
    for task in train_tasks:
        dataloaders = {"train": task.loader("train", batch_size=batch_size), "val": task.loader("val")}
        all_data_loaders.append(dataloaders)
        loss_fns.append(task.loss_fn())

    init_params = deepcopy(model.state_dict())
    while iters < num_iters_outer:
        delta_params = {name: 0 for name in init_params}
        avg_total_loss = 0

        for data_loaders, loss_fn in zip(all_data_loaders, loss_fns):
            if iters > num_iters_outer:
                break

            # inner train step
            total_loss = maml_inner_train(model, optimizer, 
                                        data_loaders, loss_fn, 
                                        device, num_iters_inner)
            avg_total_loss += total_loss
            wandb.log({'per_task_loss': total_loss})
            
            # re-load initial params/meta weights
            model.load_state_dict({name: init_params[name] for name in init_params})

            # Update initial parameters using update_parameters function
            delta_params_ = maml_update_parameters(model, init_params, total_loss)
            
            # Save delta for add at end of meta batch
            delta_params = {name:delta_params[name] + delta_params_[name] for name in delta_params}
            
            iters += 1

        # Update meta parameters after batch is complete
        init_params = {name:delta_params[name] + init_params[name] for name in delta_params}
        avg_total_loss /= len(all_data_loaders)
        wandb.log({'average_loss_over_tasks': avg_total_loss})

        # Start of validation tasks
        model.load_state_dict({ name: init_params[name] for name in init_params})
        with torch.no_grad():
            val_loss = 0
            for data_loaders, loss_fn in zip(val_data_loaders, val_loss_fns):
                val_loss += util.get_meta_loss_on_dataloader(model, data_loaders, loss_fn, num_iters_inner)

        val_loss /= len(val_data_loaders)
        self.wandb_run.log({"Validation Loss": val_loss})

        # Save model if better
        if val_loss < min_val_loss:
            print(f"Saving model...")
            model_io.save_model_checkpoint(save_to, model)
            min_val_loss = val_loss


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return init_params


def reptile_inner_train_loop(model, optimizer, data_loaders, loss_fn, device, num_iters):
    """Inner training loop for meta training procedure.

    Args:
        model (torch.nn.Module): model to train
        optimizer (torch.optim): optimizer
        data_loaders (dict): dictionary containing "train": train data loader 
        loss_fn (torch.function): loss function
        device (torch.device): gpu if enabled
        num_iters (int): number of iterations to train on
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

            total_loss += loss
            iters += 1

    return total_loss

def reptile_update_params(model, init_params, new_weights):
    """Performs the reptile meta update step

    Reptile update step is as follows:
        1. Take the average of the "new weights" each time we inner train on a task, we get a set of new weights
        2. meta weights = (meta weights - new weights) * meta learning rate

    Args:
        model (nn.module): Model we are training on
        init_params (state_dict): Meta parameters/ initial parameters used for the inner step
        new_weights (arr): Array of state_dicts. Updated parameters from inner training loop. One state dict per task trained on.
        total_loss (Tensor.float): Total loss from meta train step

    Returns:
        [type]: [description]
    """
    # setup to average weights
    ws = len(new_weights)
    fweights = { name : new_weights[0][name]/float(ws) for name in new_weights[0] }

    # average all of the fast weights/new weights
    for i in range(1, ws):
        for name in new_weights[i]:
            fweights[name] += new_weights[i][name]/float(ws)

    # do reptile step
    updated_meta_weights = {name : 
        init_params[name] + ((fweights[name] - init_params[name]) * meta_lr) for name in init_params}
    
    return updated_meta_weights

def meta_outer_train_loop(model, optimizer, train_tasks, 
                     validation_tasks, batch_size,
                     num_iters_inner, num_iters_outer,
                     update_parameters_inner, update_parameters,
                     device, save_to):
    """Meta train a model

    Args:
        model (nn.Module): Initialized model to meta-train.
        optimizer (torch.optim): Optimizer to train inner loop.
        train_tasks (arr): Array of Task objects. 
        validation_tasks (arr): Array of Task objects for validation.
        batch_size (int): batch size for data loaders.
        num_iters_inner (int): Number of training iterations in the inner loop. 
                               One step = one parameter update in the inner loop.
        num_iters_outer (int): Number of outer steps. How many times the init parameters are updated.
        update_parameters_inner: Function for the inner update step. MAML requires a different inner training loop
        update_parameters (function): Function to update the init params. This should take in fn(model, init_params, new_params)
        device (torch.device): Device to train on. 
        save_to (str): Name of save file. Will save model with best validation accuracy.

    Returns:
        nn.parameter: Final initialization parameters.
    """
    since = time.time()
    iters = 0
    min_val_loss = 10000

    # Turn tasks into data loaders + loss fns
    all_data_loaders = []
    loss_fns = []
    for task in train_tasks:
        dataloaders = {"train": task.loader("train", batch_size=batch_size), "val": task.loader("val")}
        all_data_loaders.append(dataloaders)
        loss_fns.append(task.loss_fn())

    while iters < num_iters_outer:
        init_params = deepcopy(model.state_dict())
        new_weights = []
        avg_total_loss = 0

        for data_loaders, loss_fn in zip(all_data_loaders, loss_fns):
            if iters > num_iters_outer:
                break

            # inner train step
            total_loss = update_parameters_inner(model, optimizer, 
                                                data_loaders, loss_fn, 
                                                device, num_iters_inner)
            avg_total_loss += total_loss
            wandb.log({'per_task_loss': total_loss})
            
            # save a copy of fast weights/new weights
            new_weights.append(deepcopy(model.state_dict()))
            
            # re-load initial params/meta weights
            model.load_state_dict({name: init_params[name] for name in init_params})

            iters += 1

        # Update initial parameters using update_parameters function
        init_params = update_parameters(model, init_params, new_weights, total_loss, update_param_kwargs)

        avg_total_loss /= len(all_data_loaders)
        wandb.log({'average_loss_over_tasks': avg_total_loss})

        # Start of validation tasks
        model.load_state_dict({ name: init_params[name] for name in init_params})
        with torch.no_grad():
            val_loss = 0
            for data_loaders, loss_fn in zip(val_data_loaders, val_loss_fns):
                val_loss += util.get_meta_loss_on_dataloader(model, data_loaders, loss_fn, num_iters_inner)

        val_loss /= len(val_data_loaders)
        self.wandb_run.log({"Validation Loss": val_loss})

        # Save model if better
        if val_loss < min_val_loss:
            print(f"Saving model...")
            model_io.save_model_checkpoint(save_to, model)
            min_val_loss = val_loss

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return init_params
