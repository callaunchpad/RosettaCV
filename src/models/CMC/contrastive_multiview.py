"""
This file implements contrastive multiview coding with our additions
"""
import torch
import itertools
import random

import torch.nn as nn
import utils.model_io as model_io
import utils.data_io as data_io

from typing import TypeVar, List, Callable
from trainer.trainer import Trainer
from collections import deque
from losses.cmc_losses import get_cmc_loss_on_dataloader, get_positive_and_negative_samples

T = TypeVar("T")


class View:
    """
    An abstract container for a view with encoder, and possible decoder
    """
    get_new_id = itertools.count().__next__

    def __init__(self, encoder: nn.Module, decoder: nn.Module = None, view_id: str = ""):
        """
        Sets up the view with the given parameters
        :param encoder: The encoder to use to transform this view to the shared latent
        :param decoder: The decoder to use to transform the latent into this view
        """
        self.encoder = encoder
        self.decoder = decoder

        # Assign a numerical view_id if none is given
        self.view_id = view_id if view_id != "" else View.get_new_id()

    def encode(self, sample: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """
        Encodes into the latent space
        :param sample: The sample to encode
        :param eval_mode: Whether or not to track gradients or simply forward pass
        :return: The latent space encoding
        """
        if not eval_mode:
            return self.encoder(sample)

        is_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            latent_encoding = self.encoder(sample)

        self.encoder.train(is_training)

        return latent_encoding

    def decode(self, latent_encoding: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """
        Decodes the latent encoding back into this view
        :param latent_encoding: The latent encoding of the sample we're trying to recover
        :param eval_mode: Whether or not to track gradients/statistics or simply forward pass
        :return: The decoded latent space view of the sample
        """
        assert self.decoder is not None, "Decode method not specified for this view"

        if not eval_mode:
            return self.decoder(latent_encoding)

        is_training = self.decoder.training
        self.decoder.eval()

        with torch.no_grad():
            recovered_sample = self.decoder(latent_encoding)

        self.encoder.train(is_training)

        return recovered_sample


class WrapperModel(nn.Module):
    """
    Wraps a series of view models for easier object oriented access
    """
    def __init__(self, views: List[View], latent_dim: int, memory_bank_size: int = 200):
        super(WrapperModel, self).__init__()

        # Assign views and register submodules
        self.views = views
        self.view_encoders = nn.ModuleList([view.encoder for view in views])
        self.view_decoders = nn.ModuleList([view.decoder for view in views if view.decoder is not None])

        # Build the memory bank to sample from
        self.memory_bank = deque()
        self.memory_bank.extend([rand_vec.view(-1, latent_dim) for rand_vec
                                 in torch.randn((memory_bank_size, latent_dim))])

    def forward(self, X: List[torch.Tensor], views: List[int] = None, no_cache: bool = False) -> List[torch.Tensor]:
        """
        Performs a forward pass through the selected view models
        :param X: A list of views to encode
        :param views: A list of indices for which views to encode (if given a subset)
        :param no_cache: Whether or not to skip caching the encodings in the memory bank
        :return: A list of the encoded views
        """

        if views is None:
            views = []
        if not views:
            views = list(range(len(self.views)))

        # Encode the given views
        encoded_views = [self.views[views[i]].encode(X[i]) for i in views]

        # Enqueue these encodings in the memory bank
        if not no_cache:
            self.memory_bank.extend([view.detach() for view in encoded_views])

        return encoded_views

    def decode(self, encoding: List[torch.Tensor], views: List[int]) -> List[torch.Tensor]:
        """
        Decodes the latent embeddings of the selected views
        :param encoding: The encodings of the given views
        :param views: A list of indices for which views to decode to
        :return: A list of decoded views
        """
        # To implement after base CMC is implemented
        raise NotImplementedError()

    def get_negative_samples(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the memory bank
        :param num_samples: The number of samples to pull
        :return: The samples from the negative sample memory buffer
        """
        return torch.cat(random.sample(self.memory_bank, num_samples), 0)


class CMCTrainer(Trainer):
    """
    Class to encompass training the model

    Differences to Trainer:
        - The dataloader is assumed to yield a list of views
    """
    def train(self, epochs: int = 10, save_best: bool = True, save_to: str = None):
        """
        Overloads the training loop from the standard trainer to train via CMC
        :param epochs: The number of epochs to use
        :param save_best: Whether or not to save the model with best validation loss
        :param save_to: Where to save the models to
        :return: None
        """
        if save_to is None:
            save_to = self.wandb_run.name
        if save_best:
            with torch.no_grad():
                min_val_loss = get_cmc_loss_on_dataloader(self.model, self.train_data, self.loss_function   )

        print("Beginning training...")
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            avg_loss = 0

            # Train on all the batches
            for index, batch in enumerate(self.train_data):
                inputs = [view.to(self.device) for view in batch]

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                encodings = self.model(inputs)

                core_view, positive_samples, negative_samples = get_positive_and_negative_samples(encodings, self.model)
                loss_value = self.loss_function(core_view, positive_samples, negative_samples)

                # Backward pass
                loss_value.backward()
                self.wandb_run.log({"Train Loss": loss_value})

                self.optimizer.step()

                avg_loss += loss_value

            avg_training_loss = avg_loss / len(self.train_data)
            print(f"Average Training Loss: {avg_training_loss}")

            # Compute validation loss
            with torch.no_grad():
                val_loss = float(get_cmc_loss_on_dataloader(self.model, self.validation_data, self.loss_function))

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


from torchvision import models, datasets
from torchvision.transforms import Compose, ToTensor
from data_loader.MultiviewDatasets import get_noisy_view, MultiviewDataset, identity_view
from torch.utils.data import DataLoader
resnet_feature_size = 512

class FeatureExtractor(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.features = None

        # Register a hook for forward pass caching
        self.resnet.layer4.register_forward_hook(self.intermediate_hook)
        self.linear = nn.Linear(resnet_feature_size, latent_dim)

    def forward(self, X):
        # Perform forward pass
        self.resnet(X)

        return self.linear(self.features)

    def intermediate_hook(self, module, inputs, outputs):
        self.features = torch.clone(outputs.view(outputs.size()[0], -1))



if __name__ == "__main__":
    fe1 = FeatureExtractor()
    fe2 = FeatureExtractor()

    base_data = datasets.CIFAR10("../data", transform=Compose([ToTensor()]))
    noisy_view = get_noisy_view()

    ds = MultiviewDataset(base_data, [identity_view, noisy_view])

    train_proportion = 0.7
    train_len = int(len(ds) * 0.7 * 0.1)
    valid_len = int(len(ds) * 0.1 - train_len)
    rest = len(ds) - train_len - valid_len

    train_data, valid_data, _ = torch.utils.data.random_split(ds, [train_len, valid_len, rest])
    train_loader, valid_loader = DataLoader(train_data, batch_size=16), DataLoader(valid_data, batch_size=16)

    from losses.cmc_losses import contrastive_loss
    from models.CMC.contrastive_multiview import CMCTrainer, View, WrapperModel

    view1, view2 = View(fe1), View(fe2)
    model = WrapperModel([view1, view2], 512)

    trainer = CMCTrainer(model, contrastive_loss, train_loader, validation_data=valid_loader)

    trainer.train(50)


