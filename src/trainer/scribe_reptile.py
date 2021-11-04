import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from copy import deepcopy 

import time
#import wandb
#import model_io

# from RosettaCV repo
def get_accuracy_on_dataloader(model: nn.Module, data_loader) -> torch.Tensor:
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

# Model agnostic - you can pass in the model 
def reptile_inner_train_loop(model, optimizer, data_loader, loss_fn, device, num_iters): 
    # data_loader is specific to class 
    # we are moving the weights a little in the direction of one class 
    # data_loader = dictionary{"train": [(input, label) pairs], "validation: [(input, label) pairs]"}
        # each data loader is like a generator so it automatically increments the index 
        #   (don't just repeat on the first few samples every time)
    # device = gpu if enabled 
    iteration = 0
    total_loss = 0
    model.to(device) # update the model with the write device

    # num_iters < total number of samples because we only want to move the weights
    # a little in the direction of the class 
    for inputs, labels in data_loader['train']: # assuming inputs, labels are a tensor
        if iteration >= num_iters: 
            break 
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad() 

        with torch.set_grad_enabled(True): 
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss 
        iteration += 1
    return total_loss

def reptile_update_params(model, init_params, new_weights): 
    # parameters removed: model, total_loss
    ws = len(new_weights)
    # fast weights: divide by length to do a running average
    # averaging over new_weights[0] .... new_weights[n-1] for a specific name
    fweights = { name : new_weights[0][name]/float(ws) for name in new_weights[0] }

    for i in range(1, ws): 
        for name in new_weights[i]: 
            fweights[name] += new_weights[i][name]/float(ws)

    meta_lr = 0.2

    updated_meta_weights = {name : 
        init_params[name] + ((fweights[name] - init_params[name]) * meta_lr) for name in init_params}
    
    return updated_meta_weights

def meta_outer_train_loop(model, optimizer, train_tasks, 
                     validation_tasks, batch_size,
                     num_iters_inner, num_iters_outer,
                     inner_train_loop, update_parameters,
                     device, save_to): 
    since = time.time()
    iters = 0
    min_val_loss = float('inf')

    model.to(device)
    print(device)

    # Turn tasks into data loaders + loss fns
    all_data_loaders = []
    loss_fns = []
    for task, validation in zip(train_tasks, validation_tasks):
        dataloaders = {"train": task.loader(batch_size=batch_size), "val": validation.loader()}
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
                total_loss = inner_train_loop(model, optimizer, 
                                                    data_loaders, loss_fn, 
                                                    device, num_iters_inner)
                avg_total_loss += total_loss
                #wandb.log({'per_task_loss': total_loss})
                
                # save a copy of fast weights/new weights
                new_weights.append(deepcopy(model.state_dict()))

                # compute accuracy
                accuracy = get_accuracy_on_dataloader(model, data_loaders["val"])
                #wandb.log({'per_task_accuracy': accuracy})
                
                # re-load initial params/meta weights
                model.load_state_dict({name: init_params[name] for name in init_params})

                iters += 1 

            # Update initial parameters using update_parameters function
            init_params = update_parameters(model, init_params, new_weights)

            avg_total_loss /= len(all_data_loaders)
           # wandb.log({'average_loss_over_tasks': avg_total_loss})

            # Start of validation tasks
            model.load_state_dict({ name: init_params[name] for name in init_params})
            with torch.no_grad():
                val_loss = 0
                task_count = 0
                avg_accuracy = 0
                for validation_task in validation_tasks:
                    # Get data loaders and loss function
                    data_loaders = {"train": validation_task.loader()} # Temp fix.. else API cries
                    loss_fn = validation_task.loss_fn()

                    # Train for 32 steps, and get the validation loss from here
                    val_loss += inner_train_loop(model, optimizer, 
                                                    data_loaders, loss_fn, 
                                                    device, num_iters_inner)

                    # Also compute accuracy
                    avg_accuracy += get_accuracy_on_dataloader(model, data_loaders["train"])

                    # Re-load the model for next iteration of loop
                    model.load_state_dict({ name: init_params[name] for name in init_params})

            val_loss /= len(validation_tasks)
            avg_accuracy /= len(validation_tasks)
         #   wandb.log({"Average accuracy": avg_accuracy})
         #   wandb.log({"Validation Loss": val_loss})

            # Save model if better
            if val_loss < min_val_loss:
                print(f"Saving model...")
              #  model_io.save_model_checkpoint(save_to, model)
                min_val_loss = val_loss

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        return init_params
