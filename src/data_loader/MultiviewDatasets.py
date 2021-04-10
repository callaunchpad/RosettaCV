"""
This file defines the multiview dataset as a dataset that consumes a base
dataset and defines methods for creating views from the samples this base
dataset yields
"""
import torch

from torch.utils.data import Dataset
from typing import Callable, List

"""
Multiview Dataset
"""


class MultiviewDataset(Dataset):
    """
    Defines a dataset that transforms another dataset into a set of views
    Notes:
        -  The identity view is not a default view for either inputs or labels
    """

    def __init__(self, base_dataset: Dataset, input_views: List[Callable] = [], label_views: List[Callable] = []):
        """
        Initializes the base dataset and view creation methods
        :param base_dataset: The dataset to draw views from, assumed to yield a tuple
        (input, label) for every __getitem__ call
        :param input_views: The methods to create views from the input (e.g. view_k = input_views[k](input)
        :param label_views: The methods to create views form the label (e.g. view_k = label_views[k](label)
        """
        self.base_dataset = base_dataset
        self.input_views = input_views
        self.label_views = label_views

        assert len(input_views) + len(label_views) >= 2, "Must specify at least 2 views for CMC to work!"

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item):
        input, label = self.base_dataset[item]

        return [view_callback(input) for view_callback in self.input_views] + \
               [view_callback(label) for view_callback in self.label_views]


"""
Sample View Functions
"""


def identity_view(X: torch.Tensor) -> torch.Tensor:
    """
    Specifies the identity view, used to include the original input or label in the
    set of views for a multiview dataset
    :param X: The original input
    :return: The original input
    """
    return X


def get_noisy_view(sigma: float = 0.1) -> Callable:
    def noisy_view(X: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(X) * sigma + X

    return noisy_view
