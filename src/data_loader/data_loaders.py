import os
import gzip
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

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
        train_set, val_set = random_split(train_set, lengths)

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
