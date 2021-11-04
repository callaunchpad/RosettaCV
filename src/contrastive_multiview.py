"""
This file implements contrastive multiview coding with our additions
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import itertools
import random

import torch.nn as nn
import utils.model_io as model_io
import utils.util as util

from typing import TypeVar, List, Callable
from trainer.trainer import Trainer
from collections import deque
from losses.cmc_losses import get_cmc_loss_on_dataloader, get_positive_and_negative_samples, l2_reconstruction_loss, language_reconstruction_loss, domain_confusion_loss
from itertools import product

T = TypeVar("T")

class View:
    """
    An abstract container for a view with encoder, and possible decoder
    """
    get_new_id = itertools.count().__next__

    def __init__(self, encoder: nn.Module, decoder: nn.Module = None, view_id: str = "",
                 reconstruction_loss: Callable = None):
        """
        Sets up the view with the given parameters
        :param encoder: The encoder to use to transform this view to the shared latent
        :param decoder: The decoder to use to transform the latent into this view
        :param view_id: A string identifier e.g. Image caption
        :param reconstruction_loss: A loss for comparing latents that are decoded into this
        view
        """
        # assert not decoder is not None and reconstruction_loss is not None, "Reconstruction loss must be specified if" \
                                                                        # "decoder is specified."
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss

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

    def decode(self, latent_encoding: torch.Tensor, label: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """
        Decodes the latent encoding back into this view
        :param latent_encoding: The latent encoding of the sample we're trying to recover
        :param label: The label to decode to (for label aware decoding like BERT)
        :param eval_mode: Whether or not to track gradients/statistics or simply forward pass
        :return: The decoded latent space view of the sample
        """
        assert self.decoder is not None, "Decode method not specified for this view"

        if not eval_mode:
            if hasattr(self.decoder, "needs_labels") and self.decoder.needs_labels:
                return self.decoder(latent_encoding, label)
            else:
                return self.decoder(latent_encoding)

        is_training = self.decoder.training
        self.decoder.eval()

        with torch.no_grad():
            if hasattr(self.decoder, "needs_labels") and self.decoder.needs_labels:
                recovered_sample = self.decoder(latent_encoding, label)
            else:
                recovered_sample = self.decoder(latent_encoding)

        self.encoder.train(is_training)

        return recovered_sample

    def decodable(self) -> bool:
        """
        :return: Returns true if this view has a decoder attached
        """
        return self.decoder is not None

    def get_id(self) -> str:
        """
        Returns the ID for the given view
        """
        return self.view_id


class WrapperModel(nn.Module):
    """
    Wraps a series of view models for easier object oriented access
    """
    def __init__(self, views: List[View], latent_dim: int, memory_bank_size: int = 200):
        super(WrapperModel, self).__init__()
        assert len(views) >= 2, "Must specify at least 2 views!"

        # Assign views and register submodules
        self.latent_dim = latent_dim
        self.views = views
        self.view_encoders = nn.ModuleList([view.encoder for view in views])
        self.view_decoders = nn.ModuleList([view.decoder for view in views if view.decoder is not None])

        # Build the memory bank to sample from
        model_device = util.get_project_device()
        self.memory_bank = deque()
        self.memory_bank.extend([rand_vec.view(-1, latent_dim) for rand_vec
                                 in torch.randn((memory_bank_size, latent_dim)).to(model_device)])

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
    def __init__(self, *args, use_domain_confusion: bool = False, num_decodings_per_step: int = 0, **kwargs):
        """
        Sets up the CMCTrainer
        :param args: Args for superclass
        :param num_decodings_per_step: Number of encode-decode pairings to sample per step, default
        is not to use decodings
        :param kwargs: Keyword args for superclass
        """
        super(CMCTrainer, self).__init__(*args, **kwargs)

        self.num_decodings = num_decodings_per_step
        self.use_decoding_loss = num_decodings_per_step != 0
        self.use_domain_confusion = use_domain_confusion

        # Setup the view pairs
        all_views = self.model.views
        self.decodable_views = [view for view in all_views if view.decodable()]
        self.encode_decode_pairs = list(product(range(len(all_views)), range(len(self.decodable_views))))

    def decoding_loss(self, inputs: List[torch.Tensor], encodings: List[torch.Tensor]) -> torch.Tensor:
        """
        Samples decoding pathways and adds decoding loss to contrastive loss
        :param inputs: The inputs to the CMC encoders
        :param encodings: The encodings that were produced by the model
        :return: A loss on the decodings sampled
        """
        # Sample decoding pathways
        decodings = random.sample(self.encode_decode_pairs, k=self.num_decodings)

        # Perform all relevant decodings
        decoding_loss = torch.Tensor([0]).to(util.get_project_device())

        for encode_view_ind, decode_view_ind in decodings:
            decoded_view = self.model.views[decode_view_ind]

            # decoded_view_out = decoded_view.decode(encodings[encode_view_ind])
            decoded_view_out = decoded_view.decode(encodings[encode_view_ind], inputs[decode_view_ind])

            reconstruction_loss = decoded_view.reconstruction_loss(decoded_view_out, inputs[decode_view_ind])

            # Report this reconstruction loss
            self.wandb_run.log({f"{self.model.views[encode_view_ind].get_id()} -> "
                                f"{self.model.views[decode_view_ind].get_id()} Reconstruction Loss": reconstruction_loss})

            decoding_loss += reconstruction_loss

        return decoding_loss

    def train(self, epochs: int = 10, save_best: bool = True, save_to: str = None):
        """
        Overloads the training loop from the standard trainer to train via CMC
        :param epochs: The number of epochs to use
        :param save_best: Whether or not to save the model with best validation loss
        :param save_to: Where to save the models to
        :return: None
        """
        print(f"Training set size: {self.train_data.batch_size * len(self.train_data)}\n"
              f"Validation set size: {self.validation_data.batch_size * len(self.validation_data)}")
        
        if save_to is None:
            save_to = self.wandb_run.name
        if save_best:
            with torch.no_grad():
                min_val_loss = get_cmc_loss_on_dataloader(self.model, self.train_data, self.loss_function)
        
        print("Received loss")

        # Initialize the Domain confusion loss' DANN
        num_domains = util.get_num_domains(self.train_data)
        latent_dim = self.model.latent_dim
        domain_loss = domain_confusion_loss(latent_dim, num_domains)

        print("Beginning training...")
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            avg_loss = 0

            # Train on all the batches
            for index, inputs in enumerate(self.train_data):
                # Send to GPU if possible
                for i, view in enumerate(inputs):
                    if isinstance(view, torch.Tensor):
                        inputs[i] = view.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                encodings = self.model(inputs)

                core_view, positive_samples, negative_samples = get_positive_and_negative_samples(encodings, self.model)
                loss_value = self.loss_function(core_view, positive_samples, negative_samples)

                if self.use_domain_confusion:
                    total_domain_loss = domain_loss(encodings)
                    self.wandb_run.log({"Domain Confusion Loss": total_domain_loss})
                    loss_value += total_domain_loss

                if self.use_decoding_loss:
                    reconstruction_loss = self.decoding_loss(inputs, encodings)
                    self.wandb_run.log({"Contrastive Loss": loss_value})
                    loss_value += reconstruction_loss[0]
                else:
                    self.wandb_run.log({"Contrastive Loss": loss_value})

                # Backward pass
                loss_value.backward()
                self.optimizer.step()

                avg_loss += loss_value

            avg_training_loss = avg_loss / len(self.train_data)
            print(f"Average Training Loss: {avg_training_loss}")

            # Compute validation loss
            with torch.no_grad():
                val_loss = float(get_cmc_loss_on_dataloader(self.model, self.validation_data, self.loss_function))

            print(f"Average Validation Loss: {val_loss}")
            self.wandb_run.log({"Validation Loss": val_loss})

            if min_val_loss > val_loss > 0 and save_best:
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
from torchvision.transforms import Compose, ToTensor, Resize
from data_loader.MultiviewDatasets import MultiviewDataset, identity_view, get_coco_captions
from torch.utils.data import DataLoader
resnet_feature_size = 512


if __name__ == "__main__":
    from models.CMC.ResNetEncoder import ResNetEncoder
    from models.CMC.Decoder import Decoder
    from models.CMC.language_models import TextEncoder, TextDecoder

    # Ensure aspect ratio is 0.75
    image_size = (480, 640)
    device = util.get_project_device()
    latent_dim = 512

    # Define encoders, decoders, and views
    image_encoder = ResNetEncoder(device, latent_dim=latent_dim)
    image_decoder = Decoder(*image_size)
    image_view = View(image_encoder, image_decoder, "Image", reconstruction_loss=l2_reconstruction_loss)

    caption_encoder = TextEncoder(latent_dim)
    caption_decoder = TextDecoder(latent_dim)
    caption_view = View(caption_encoder, caption_decoder, "Caption", reconstruction_loss=language_reconstruction_loss)

    path2data = "/datasets/coco/data/train2017"
    path2json = "/datasets/coco/data/annotations/captions_train2017.json"

    coco_train = datasets.CocoDetection(root = path2data, annFile = path2json, transform=Compose([Resize(image_size), ToTensor()]))
    ds = MultiviewDataset(coco_train, [identity_view], [get_coco_captions])

    train_proportion = 0.7
    train_len = int(len(ds) * 0.3)
    valid_len = int(len(ds) * 0.1)
    rest = len(ds) - train_len - valid_len

    train_data, valid_data, _ = torch.utils.data.random_split(ds, [train_len, valid_len, rest])
    train_loader, valid_loader = DataLoader(train_data, batch_size=16), DataLoader(valid_data, batch_size=16)

    from losses.cmc_losses import contrastive_loss
    from models.CMC.contrastive_multiview import CMCTrainer, View, WrapperModel

    model = WrapperModel([image_view, caption_view], latent_dim)

    trainer = CMCTrainer(model, contrastive_loss, train_loader, validation_data=valid_loader, num_decodings_per_step=2, use_domain_confusion=True)
    trainer.train(500, save_best = True)



