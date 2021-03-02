import os
import gzip
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

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

def get_fashion_mnist_dataloader(batch_size):
    train_dataset = FashionMnistDataset(train=True)

    lengths = [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)]
    split_train_dataset, split_val_dataset = random_split(train_dataset, lengths)

    def loss_fn(x):
        nn.CrossEntropyLoss()

    return (DataLoader(dataset=split_train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(dataset=split_val_dataset, batch_size=batch_size, shuffle=True),
            loss_fn)

if __name__ == "__main__":
    print("[*] Testing FashionMnist")
    test_train_dataloader, test_val_dataloader, loss_fn = get_fashion_mnist_dataloader(32)
    inp, label = next(iter(test_train_dataloader))
    print(inp, label)
    inp, label = next(iter(test_val_dataloader))
    print(inp, label)
    print(loss_fn)