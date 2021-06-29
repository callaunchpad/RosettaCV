"""
This file contains methods to make a few shot dataloader out of a vanilla dataloader
"""
import utils.data_io as data_io
import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset


class SubSampledDataset(Dataset):
    """
    A dataset that subsamples the given dataset using the given indices
    """
    def __init__(self, base_dataset: Dataset, indices: np.array):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_point = self.base_dataset[self.indices[index]]

        if type(data_point) == tuple:
            return {
                'input': data_point[0],
                'label': data_point[-1]
            }

        return data_point


def get_few_shot_dataloader(dataset: Dataset, indices_file: str = None, sample_size: float = 0.01,
                            batch_size: int = 32, by_class: bool = True) -> DataLoader:
    """
    Takes in a dataset (and possibly a file specifying a list of indices)
    and creates a dataset for a subset of the given dataset
    :param dataset: The torch dataset to sub-index
    :param indices_file: [Optional] The file containing sub-indices to take, rooted at src/data
    :param sample_size: The proportion of the data to keep in the few-shot dataloader
    :param batch_size: The batch size to use for the dataloader
    :param by_class: Whether or not to match class-wise proportions from the original dataset
    set to false if the task isn't a classification task
    :return: A dataloader that samples a percentage of the dataset
    """
    assert 0 < sample_size < 1, "Sample size must be between 0 and 1"

    # Possibly load the indices
    if indices_file is not None:
        # Load a numpy array from the file
        indices = np.load(f"{data_io.get_data_path()}/{indices_file}")
    else:
        # Sample indices and save to data folder
        indices = sample_indices(dataset, sample_size, by_class=by_class)
        np.save(f"{data_io.get_data_path()}/subsampled_data_{sample_size}", indices)
    # Return a dataloader
    return DataLoader(SubSampledDataset(dataset, indices), batch_size=batch_size, shuffle=True)


def sample_indices(dataset: Dataset, sample_size: float = 0.1, by_class: bool = True) -> np.array:
    """
    Samples indices from a given dataset to preserve <sample_size> * len(dataset) amount of data
    :param dataset: The dataset to sample from
    :param sample_size: The proportion of the datset to keep
    :return: A numpy array containing the indices for sampling
    """
    # If not by class, simply sample indices at random
    if not by_class:
        return np.random.randint(0, len(dataset), size=int(sample_size * len(dataset)))

    # Loop over the dataloader and get all the indices for different classes
    class_to_index_list = {}

    for index in range(len(dataset)):
        data_point = dataset[index]

        if type(data_point) == tuple:
            label = data_point[-1]
        else:
            label = data_point['label']

        class_to_index_list[label] = class_to_index_list.get(label, []) + [index]

    # Sample from these classes
    return_indices = []
    for label in class_to_index_list:
        return_indices.append(np.random.choice(class_to_index_list[label],
                                          size=int(len(class_to_index_list[label]) * sample_size)))

    return np.concatenate(return_indices, axis=0)
