"""
An object oriented wrapper for the training process
"""
import torch
import wandb

import torch.nn as nn
import utils.model_io as model_io
import utils.util as util

from torch.utils.data.dataloader import DataLoader
from typing import Optional, Callable, List


class Trainer:
    def __init__(self, model: nn.Module, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 train_data: DataLoader, validation_data: Optional[DataLoader] = None, optimizer: Optional = None,
                 per_epoch_callbacks: List[Callable] = [], device: torch.device = None, checkpoint_file: str = ""):
        """
        Trains a model given the data, loss function, and possibly and optimizer
        :param model: The model to train
        :param loss_function: The loss function to evaluate for training
        :param train_data: The dataloader to use for training the model
        :param validation_data: The dataloader used for model validation
        :param optimizer: [Optional] The optimizer used to update weights of the model
        :param per_epoch_callbacks: A set of callbacks to be called every epoch. Will be passed
        the model, data loaders, optimizer, and wandb run. May be used for (e.g. logging images)
        :param device: The device ((C/G/T)PU) to train and validate on
        :param checkpoint_file: The location of the saved run to load the model and optimizer from
        """
        # Create optimizer if none passed
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=0.001)
        else:
            self.optimizer = optimizer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create a learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # Save the other class parameters
        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.train_data = train_data
        self.validation_data = validation_data
        self.callbacks = per_epoch_callbacks

        # Load the model if checkpoint is given
        if checkpoint_file != "":
            model_save_dict, optimizer_save_dict = model_io.load_model_checkpoint(checkpoint_file)

            self.model.load_state_dict(model_save_dict)

            if optimizer_save_dict is not None:
                self.optimizer.load_state_dict(optimizer_save_dict)

        # Initialize wandb and watch model
        self.wandb_run = wandb.init(project='RosettaCV', entity='cal-launchpad')
        wandb.watch(self.model)

    def train(self, epochs: int = 10, save_best: bool = True, save_to: str = None):
        """
        Trains the model defined in the trainer
        :param epochs: The number of loops over the entire dataset to use
        :param save_best: A flag of whether or not to save the best model by validation loss
        :param save_to: The location to save the best model to (rooted at src/)
        :return: None
        """
        if save_to is None:
            save_to = self.wandb_run.name

        if save_best:
            with torch.no_grad():
                min_val_loss = util.get_loss_on_dataloader(self.model, self.validation_data, self.loss_function)

        print("Beginning training...")
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            avg_loss = 0

            # Train on all the batches
            for index, batch in enumerate(self.train_data):
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                predicted_values = self.model(inputs)

                if labels.shape != predicted_values.shape and epoch == 0 and index == 0:
                    wandb.termwarn("Labels and values have different shape")

                # Backward pass
                loss_value = self.loss_function(predicted_values, labels)
                loss_value.backward()
                self.wandb_run.log({"Train Loss": loss_value})

                self.optimizer.step()

                avg_loss += loss_value

            avg_training_loss = avg_loss / len(self.train_data)
            print(f"Average Training Loss: {avg_training_loss}")

            # Compute validation loss
            with torch.no_grad():
                val_loss = util.get_loss_on_dataloader(self.model, self.validation_data, self.loss_function)

            print(f"Average Validation Loss: {val_loss}")
            self.wandb_run.log({"Validation Loss": val_loss})

            if val_loss < min_val_loss and save_best:
                print(f"Validation loss of {val_loss} better than previous best of {min_val_loss}")
                print(f"Saving model...")
                model_io.save_model_checkpoint(save_to, self.model, self.optimizer)
                min_val_loss = val_loss

            # Step the LR scheduler
            self.scheduler.step(val_loss)

            # Call the per epoch callbacks
            for callback in self.callbacks:
                callback(self.model, self.train_data, self.validation_data, self.optimizer, self.wandb_run)

        self.wandb_run.alert("Run Finished", f"Your run ({self.wandb_run.name}) finished\nValidation Loss: {val_loss}")
        self.wandb_run.finish(0)