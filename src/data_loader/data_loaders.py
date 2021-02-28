import os
import gzip
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

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

class FashionMnist():
    def __init__(self, batch_size=32, path="/datasets/fashion_mnist", transform=None, shuffle=True):
        self.batch_size = batch_size
        self.path = path
        self.transform = transform
        self.shuffle = shuffle

    def get_train_dataloader(self):
        train_dataset = FashionMnistDataset(train=True, path=self.path, transform=self.transform)
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_test_dataloader(self):
        test_dataset = FashionMnistDataset(train=True, path=self.path, transform=self.transform)
        return DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

if __name__ == "__main__":
    print("[*] Testing FashionMnist")
    testFashionMnist = FashionMnist()
    testTrainDataLoader = testFashionMnist.get_train_dataloader()
    inp, label = next(iter(testTrainDataLoader))
    print(inp, label)