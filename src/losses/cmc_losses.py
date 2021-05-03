"""
This file defines the (possibly many) losses for our variation of CMC

Reference paper: https://arxiv.org/pdf/1906.05849.pdf
"""
import torch

import torch.nn as nn
import utils.util as util
import torch.nn.functional as F

from typing import Callable, Tuple, List, Union
from torch.utils.data import DataLoader
from transformers import BertTokenizer

"""
CONSTANTS 
"""
PAD_TOKEN = 0
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
mse = torch.nn.MSELoss()


def get_positive_and_negative_samples(encodings: List[torch.Tensor], model: nn.Module,
                                      num_negative_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gets the positive and negative samples for contrastive loss
    :param encodings: The encodings to use in the contrastive loss
    :param model: The model that produced the encodings, included here to sample
    from the model's memory bank
    :param num_negative_samples: The number of negative samples to contrast against
    in the contrastive loss
    :return: A tuple containing (encodings, positive_samples, negative_samples)
    See contrastive_loss below for the shapes of these tensors
    """
    num_views = len(encodings)
    batch_size = encodings[0].size()[0]

    # Get the core view of shape N x D and reshape to N x 1 x D
    core_views = encodings[0]
    core_views = core_views.unsqueeze(1)

    # Sample a view for each core view to contrast with
    positive_views_indices = torch.randint(1, num_views, (batch_size,))
    positive_views = torch.cat([
        encodings[positive_views_indices[i]][i].unsqueeze(0) for i in range(batch_size)
    ]).unsqueeze(1)

    # Sample negative views to contrast against
    negative_views = model.get_negative_samples(num_negative_samples).unsqueeze(0).tile((batch_size, 1, 1))

    return core_views, positive_views, negative_views


def get_cmc_loss_on_dataloader(model: nn.Module, dataloader: DataLoader, loss_fn: Callable) -> torch.Tensor:
    """
    Gets the loss on a dataloader with a different assumed loss signature to that in util.util
    :param model: The model to evaluate on
    :param dataloader: The dataloader to evaluate on
    :param loss_fn: The loss metric to use
    :return: The loss function applied to the dataloader
    """
    # The device the model is on
    device = util.get_project_device()
    total_loss = torch.FloatTensor([0]).to(device)

    # Get all the encodings
    for batch in dataloader:
        # Send to GPU if available
        for i, view in enumerate(batch):
            if isinstance(view, torch.Tensor):
                batch[i] = view.to(device)

        # Get the positive and negative samples from the batch
        encodings = model(batch, no_cache=True)
        enc, pos, neg = get_positive_and_negative_samples(encodings, model)

        total_loss += loss_fn(enc, pos, neg)

    return total_loss / len(dataloader)


def contrastive_loss(encoding: torch.Tensor, positive_sample: torch.Tensor,
                     negative_samples: torch.Tensor) -> torch.Tensor:
    """
    Implements the contrastive loss defined in the CMC paper
    Defined for batch size N, embedding size D, and num_negative_samples K
    :param encoding: The encoded vector to contrast against samples (N x 1 x D)
    :param positive_sample: The positive sample to contrast against (N x 1 X D)
    :param negative_samples: The negative sample to contrast against (N x K x D)
    :return: The contrastive loss (softmax with correct label in positive location)
    """

    assert len(encoding.size()) == 3, "Expecting encoding shape: (N x 1 x D)"
    assert len(positive_sample.size()) == 3, "Expecting positive sample shape: (N x 1 x D)"
    assert len(negative_samples.size()) == 3, "Expecting negative sample shape: (N x K x D)"

    # Stack positive sample on top of negative
    all_samples = torch.cat([positive_sample, negative_samples], dim=1)

    # Compute the "critic" scores from 3.2 Implementing the Critic
    scores = torch.bmm(all_samples, torch.transpose(encoding, 1, 2)).squeeze(-1)

    # Compute the contrastive loss
    targets = torch.zeros(scores.size()[0]).long().to(util.get_project_device())

    return F.cross_entropy(scores, targets)


def language_reconstruction_loss(predicted_logits: torch.Tensor, ground_truth_caption: Union[str, List]) -> torch.Tensor:
    """
    The language reconstruction loss, to compare a generated caption to a ground truth
    :param predicted_logits: The logits over the vocabulary predicted by the language decoder model
    :param ground_truth_caption: The actual caption for the image
    :return: The cross entropy loss of the caption and the predicted values
    """
    # Tokenize the caption into BERT's vocabulary
    if type(ground_truth_caption) == str:
        ground_truth_tok = tokenizer.encode(ground_truth_caption, padding=True, return_tensors="pt")
    else:
        ground_truth_tok = tokenizer.batch_encode_plus(ground_truth_caption, padding=True, return_tensors="pt")['input_ids']

    # Reshape ground truth and predicted
    predicted_logits = predicted_logits.view(-1, predicted_logits.shape[-1])  # Shape[-1] is the vocab size
    ground_truth_tok = ground_truth_tok.view(-1)

    # Cross entropy ignoring locations that are padded
    return F.cross_entropy(predicted_logits, ground_truth_tok, ignore_index=PAD_TOKEN)


def l2_reconstruction_loss(decoded_view, inputs):
    return mse(decoded_view, inputs)
