"""
This file implements contrastive multiview coding with our additions
"""
import torch
import itertools

import torch.nn as nn

from typing import List
from typing import TypeVar
from torch.utils.data import DataLoader

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

        return latent_encoding


def train_cmc(views: List[View], view_loader: DataLoader, num_negatives: int = 100):
    """
    Trains the models specified by the views via a CMC approach
    :param views: The views to be used in the training procedure
    :param view_loader: The dataloader for the views, assumed to yield a list with length
    equal to the number of views
    :param num_negatives: The number of negative samples to sum over in the contrastive loss
    :return: None
    """


"""
Helpers
"""

if __name__ == "__main__":
    exit(0)