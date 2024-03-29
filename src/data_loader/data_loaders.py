import os
import glob
import numpy as np
import torch
import torchvision

import utils.data_io as data_io

from pathlib import Path
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from data_loader.few_shot_dataloaders import get_few_shot_dataloader
from typing import List

class Task:
    
    def loader(self, **kwargs):
        """
        Get Pytorch DataLoader corresponding to task data.
        """
        raise NotImplemented

    def loss_fn(self):
        raise NotImplemented

class ImageNetTask(Task):

    # override this for tasks involving pre-processing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    def loader(self, train: bool, **kwargs):
        """
        train: whether to get loader for train or test dataset
        kwargs: arguments for DataLoader
        NOTE: relies on installation on CSUA server
        """
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 32

        if train:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/ilsvrc2012/train/',
                                                           transform=self.transform), shuffle=True, batch_size=self.batch_size, **kwargs)
        else:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/ilsvrc2012/val/',
                                                           transform=self.transform), shuffle=True, batch_size=self.batch_size, **kwargs)

class ImageNetClass(ImageNetTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss()

class ImageNetDenoising(ImageNetTask):
    def loss_fn(self):
        return torch.nn.MSELoss()


class MiniImageNetTask(Task):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    def loader(self, train: bool, **kwargs):
        """
        train: true for train dataset, false for test dataset
        kwargs: DataLoader arguments (eg. batch_size)
        """
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 32

        # transform = transforms.Compose([transforms.ToTensor(),
        #                         transforms.Normalize((0.1307,), (0.3081,)),
        #                         ])

        # lengths = [int(len(data_set)*0.8), int(len(data_set)*0.2)]
        # train_set, val_set = random_split(data_set, lengths)

        if train:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/imagenet-1000/train/',
                                                           transform=self.transform), shuffle=True, batch_size=self.batch_size, **kwargs)
        else:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/imagenet-1000/val/',
                                                           transform=self.transform), shuffle=True, batch_size=self.batch_size, **kwargs)

class MiniImageNetClass(MiniImageNetTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss()

class MiniImageNetDenoising(MiniImageNetTask):
    def loss_fn(self):
        return torch.nn.MSELoss()


class OmniglotTask(Task):
    def loader(self, train: bool, **kwargs):
        """
        train: true if for train dataset, false if for test dataset
        """
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 32
        
        data = torchvision.datasets.Omniglot(
            root="/datasets/", background = train, download=True, transform=torchvision.transforms.ToTensor()
        ) 
            
        dataloader = torch.utils.data.DataLoader(train, batch_size = self.batch_size, shuffle = True, num_workers = 2)

        return dataloader

class OmniglotClass(OmniglotTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss()

class OmniglotFewShot(OmniglotClass):
    # Override the loader function to incorporate get_few_shot_dataloader.
    def loader(self, train: bool, **kwargs):
        """
        train: true if for train dataset, false if for test dataset
        """
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 32

        raw_data = torchvision.datasets.Omniglot(
            root="./datasets", background=train, download=True, transform=torchvision.transforms.ToTensor()
        )
        print(len(raw_data))

        dataloader = get_few_shot_dataloader(raw_data, batch_size=self.batch_size, by_class=True)

        return dataloader

class MiniImageNetByClass:
    class LabelCorrectingDataset(Dataset):
        """
        This dataset corrects labels to be 0 indexed and continuous if we are
        using imagefolder with a valid_file mask
        """

        def __init__(self, base_dataset: Dataset):
            """
            Sets up the dataset
            :param base_dataset: The base dataset to wrap
            """
            self.dataset = base_dataset

            # Construct the new label map
            labels = set()
            for _, label in self.dataset:
                labels.add(label)

            labels = list(labels)
            labels.sort()

            self.label_map = {}
            for i in range(len(labels)):
                self.label_map[labels[i]] = i

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            """
            Gets the nth item of the dataset and rearranges the labels
            :param index: The index to get
            :return: The nth item of the datset
            """
            img, label = self.dataset[index]
            label = type(label)(self.label_map[int(label)])

            return img, label

    class MiniImageNetTask(Task):
        def __init__(self, task_folder: str, start_idx: int = 0, output_width: int = 100):
            """
            Initializes the task
            :param dataset: The sub-dataset to pull the images from
            :param output_width: The max number of classes to use
            """
            print("Starting a task")
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((128, 128))
            ])

            if output_width is None:
                self.dataset = torchvision.datasets.ImageFolder(task_folder, transform=transforms)
                return

            # If we specify a number of classes, use the classes from start to output width
            first_n = set(glob.glob(f"{task_folder}/*/")[start_idx:output_width])

            def valid_file(file_path: str) -> bool:
                path = Path(file_path)

                return f"{str(path.parent.absolute())}/" in first_n

            # Create a dataset that only selects from the given n characters
            self.dataset = MiniImageNetByClass.LabelCorrectingDataset(torchvision.datasets.ImageFolder(task_folder, is_valid_file=valid_file, transform=transforms))

        def loader(self, batch_size: int = 32, **kwargs):
            return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, **kwargs, shuffle=True)

        def loss_fn(self):
            return torch.nn.CrossEntropyLoss()

    def __init__(self, num_train_classes: int = 70, num_validation_classes: int = 30,
                 items_per_class: int = 20, force_constant_size: bool = False):
        """
        Creates a class that wraps a list of tasks by sampling alphabets from the mini-imagenet dataset
        :param num_train_classes: The number of training alphabets to sample as tasks
        :param num_validation_classes: The number of validation alphabets to sample as tasks
        :param items_per_class: The number of characters to take per alphabet, if None, take them all
        :param force_constant_size: Whether or not to force the alphabets to all have the same size by subsampling
        set to min(items_per_class, 1) because 1 is the output of boolean on/off.
        """
        assert num_train_classes <= 1000, "Only 600 classes available for Mini-Imagenet train"
        assert num_validation_classes <= 100, "Only 100 validation alphabets available for Mini-Imagenet"
        assert not force_constant_size or items_per_class <= 20, "If forcing all alphabets to the same size" \
                                                                         "you must make that size <items_per_class>" \
                                                                         "less than or equal to 14"

        # List of possible train tasks
        imagenet_path = "/datasets/imagenetwhole/ilsvrc2012"
        available_train_classes = glob.glob(f"{imagenet_path}/train/*")
        available_validation_classes = glob.glob(f"{imagenet_path}/val/*")

        # Randomly choose train and validation alphabets
        train_classes = np.random.choice(available_train_classes, size=num_train_classes).tolist()
        validation_classes = np.random.choice(available_validation_classes, size=num_validation_classes).tolist()

        self.train_tasks = self.get_tasks("/datasets/imagenetwhole/ilsvrc2012/train", train_classes, force_constant_size, items_per_class)
        self.validation_tasks = self.get_tasks("/datasets/imagenetwhole/ilsvrc2012/val", validation_classes, force_constant_size, items_per_class)

    def get_tasks(self, task_folder: str, tasks: List[str], constant_size: bool = False,
                  items_per_class: int = None) -> List[Task]:
        """
        Turns lists of directories into lists of abstract tasks
        :param task_folders: The folder from which to make the task
        :param constant_size: Whether or not to force the tasks to have constant output size
        :param items_per_class: The size to force all the takss to output, see above
        :return: A list of tasks built from the folder
        """
        if constant_size:
            return [MiniImageNetByClass.MiniImageNetTask(task_folder, i, i + items_per_class) for i in range(0, 101 - items_per_class, items_per_class)]
        return [MiniImageNetByClass.MiniImageNetTask(task_folder, i, i + 20) for i in range(0, 81, 20)]

    def get_train_tasks(self) -> List[Task]:
        return self.train_tasks

    def get_validation_tasks(self) -> List[Task]:
        return self.validation_tasks



class OmniglotByAlphabet:
    class LabelCorrectingDataset(Dataset):
        """
        This dataset corrects labels to be 0 indexed and continuous if we are
        using imagefolder with a valid_file mask
        """

        def __init__(self, base_dataset: Dataset):
            """
            Sets up the dataset
            :param base_dataset: The base dataset to wrap
            """
            self.dataset = base_dataset

            # Construct the new label map
            labels = set()
            for _, label in self.dataset:
                labels.add(label)

            labels = list(labels)
            labels.sort()

            self.label_map = {}
            for i in range(len(labels)):
                self.label_map[labels[i]] = i

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            """
            Gets the nth item of the dataset and rearranges the labels
            :param index: The index to get
            :return: The nth item of the datset
            """
            img, label = self.dataset[index]
            label = type(label)(self.label_map[int(label)])

            return img, label

    class OmniglotAlphabetTask(Task):
        def __init__(self, task_folder: str, output_width: int = None):
            """
            Initializes the task
            :param task_folder: The folder to pull the alphabet from
            :param output_width: The max number of classes to use
            """
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

            if output_width is None:
                self.dataset = torchvision.datasets.ImageFolder(task_folder, transform=transforms)
                return

            # If we specify a number of classes, use the first n classes
            first_n = set(glob.glob(f"{task_folder}/*/")[:output_width])

            def valid_file(file_path: str) -> bool:
                path = Path(file_path)

                return f"{str(path.parent.absolute())}/" in first_n

            # Create a dataset that only selects from the first n characters
            self.dataset = OmniglotByAlphabet.LabelCorrectingDataset(torchvision.datasets.ImageFolder(task_folder,
                                                                                                      is_valid_file=valid_file,
                                                                                                      transform=transforms))

        def loader(self, batch_size: int = 32, **kwargs):
            return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, **kwargs, shuffle=True)

        def loss_fn(self):
            return torch.nn.CrossEntropyLoss()

    def __init__(self, num_train_classes: int = 20, num_validation_alphabets: int = 10,
                 characters_per_alphabet: int = None, force_constant_size: bool = False):
        """
        Creates a class that wraps a list of tasks by sampling alphabets from the omniglot dataset
        :param num_train_classes: The number of training alphabets to sample as tasks
        :param num_validation_alphabets: The number of validation alphabets to sample as tasks
        :param characters_per_alphabet: The number of characters to take per alphabet, if None, take them all
        :param force_constant_size: Whether or not to force the alphabets to all have the same size by subsampling
        set to min(characters_per_alphabet, 14) because 14 is the minimum characters in an alphabet
        """
        assert num_train_classes <= 30, "Only 30 alphabets available for Omniglot train"
        assert num_validation_alphabets <= 20, "Only 20 validation alphabets available for Omniglot"
        assert not force_constant_size or characters_per_alphabet <= 14, "If forcing all alphabets to the same size" \
                                                                         "you must make that size <characters_per_alphabet>" \
                                                                         "less than or equal to 14"

        # Download the dataset if necessary
        torchvision.datasets.Omniglot(root=f"{data_io.get_data_path()}", download=True, background=True)  # Train
        torchvision.datasets.Omniglot(root=f"{data_io.get_data_path()}", download=True, background=False)  # Validation

        # List of possible train tasks
        omniglot_path = f"{data_io.get_data_path()}/omniglot-py"
        available_train_alphabets = glob.glob(f"{omniglot_path}/images_background/*/")
        available_validation_alphabets = glob.glob(f"{omniglot_path}/images_evaluation/*/")

        # Randomly choose train and validation alphabets
        train_alphabets = np.random.choice(available_train_alphabets, size=num_train_classes).tolist()
        validation_alphabets = np.random.choice(available_validation_alphabets, size=num_validation_alphabets).tolist()

        self.train_tasks = self.get_tasks(train_alphabets, force_constant_size, characters_per_alphabet)
        self.validation_tasks = self.get_tasks(validation_alphabets, force_constant_size, characters_per_alphabet)

    def get_tasks(self, task_folders: List[str], constant_size: bool = False,
                  characters_per_alphabet: int = None) -> List[Task]:
        """
        Turns lists of directories into lists of abstract tasks
        :param task_folders: The folder from which to make the task
        :param constant_size: Whether or not to force the tasks to have constant output size
        :param characters_per_alphabet: The size to force all the takss to output, see above
        :return: A list of tasks built from the folder
        """
        # So finally, we return a list of Tasks with their own dataset that is supposed to be different
        # alphabets in each dataset.
        if constant_size:
            return [OmniglotByAlphabet.OmniglotAlphabetTask(folder, characters_per_alphabet) for folder in task_folders]
        return [OmniglotByAlphabet.OmniglotAlphabetTask(folder) for folder in task_folders]

    def get_train_tasks(self) -> List[Task]:
        return self.train_tasks

    def get_validation_tasks(self) -> List[Task]:
        return self.validation_tasks


class OmniglotDenoising(OmniglotTask):
    def loss_fn(self):
        return torch.nn.MSELoss()


class FashionMnistDataset(Dataset):
    # available on latte but if not, download from https://github.com/zalandoresearch/fashion-mnist and gunzip
    def __init__(self, train, path="/datasets/fashion_mnist", transform=None):
        self.path = path
        self.transform = transform
        if train:
            self.labels_path = os.path.join(self.path, "train-labels-idx1-ubyte")
            self.images_path = os.path.join(self.path, "train-images-idx3-ubyte")
        else:
            self.labels_path = os.path.join(self.path, "t10k-labels-idx1-ubyte")
            self.images_path = os.path.join(self.path, "t10k-images-idx3-ubyte")

        with open(self.labels_path, "rb") as label_file:
            self.labels = np.frombuffer(label_file.read(), dtype=np.uint8, offset=8)
        with open(self.images_path, "rb") as image_file:
            self.images = np.frombuffer(image_file.read(), dtype=np.uint8, offset=16).reshape(len(self.labels), 784)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return torch.Tensor(np.array(image)), label

class FashionMnistTask(Task):
    """
    train: true if for train dataset, false if for validation dataset
    """
    def loader(self, train: bool, **kwargs):
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 32

        train_dataset = FashionMnistDataset(train=True)

        lengths = [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)]
        split_train_dataset, split_val_dataset = random_split(train_dataset, lengths)

        if train:
            return DataLoader(dataset=split_train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(dataset=split_val_dataset, batch_size=self.batch_size, shuffle=True)

class FashionMnistClass(FashionMnistTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss()

class FashionMnistDenoising(FashionMnistTask):
    def loss_fn(self):
        return torch.nn.MSELoss()

class MNISTTask(Task):
    def loader(self, train: bool, **kwargs):
        """
        train: true for train dataset loader, false for validation dataset loader
        kwargs: arguments for DataLoader
        """
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 32

        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                ])

        data_set = datasets.MNIST('/datasets/mnist', download=True, train=True) # transform=transform
        lengths = [int(len(data_set)*0.8), int(len(data_set)*0.2)]
        train_set, val_set = random_split(data_set, lengths)

        if train:
            return DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True)
        


class MNISTClass(MNISTTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss()

class MNISTDenoising(MNISTTask):
    def loss_fn(self):
        return torch.nn.MSELoss()

if __name__ == "__main__":
    print("[*] Testing FashionMnist")
    fashion_mnist_denoise = FashionMnistDenoising()
    test_train_dataloader = fashion_mnist_denoise.loader(True)
    inp, label = next(iter(test_train_dataloader))
    print(inp, label)

    print("[*] Testing ImageNet")
    inet_denoising = ImageNetDenoising()
    loader = inet_denoising.loader(train=True, batch_size=32)
    for images, labels in loader:
        print(images.shape, labels.shape)
    
    print("[*] Testing MNIST")
    mnist_denoising = MNISTDenoising()
    loader = mnist_denoising.loader(train=True, batch_size=32)
    for images, labels in loader:
        print(images.shape, labels.shape)
