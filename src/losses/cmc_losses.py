"""
This file defines the (possibly many) losses for our variation of CMC

Reference paper: https://arxiv.org/pdf/1906.05849.pdf
"""
import torch

"""
Constant cross_entropy method
"""
cross_entropy = torch.nn.CrossEntropyLoss()


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

    all_samples = torch.cat([positive_sample, negative_samples], dim=1)
    scores = torch.bmm(all_samples, torch.transpose(encoding, 1, 2)).squeeze()
    targets = torch.zeros(scores.size()[0]).long()
    return cross_entropy(scores, targets)
