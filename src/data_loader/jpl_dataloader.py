import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class JPLLabeledTrainDataset(Dataset):
    def __init__(self, transform = None, type = "train"):
        self.root_dir = "/datasets/msl-labeled-data-set-v2.1/msl-labeled-data-set-v2.1/images"
        if (type == "train"):
            self.data = open("/datasets/msl-labeled-data-set-v2.1/msl-labeled-data-set-v2.1/train-set-v2.1.txt").readlines()
        elif (type == "valid"):
            self.data = open("/datasets/msl-labeled-data-set-v2.1/msl-labeled-data-set-v2.1/val-set-v2.1.txt").readlines()
        else:#unsure if contains labels, may not work
            print("this should not be executing yet")
            self.data = open("/datasets/msl-labeled-data-set-v2.1/msl-labeled-data-set-v2.1/test-set-v2.1.txt").readlines()
        self.data = np.array([x.split() for x in self.data])

        self.labels  = self.data[:,1]
        self.image_names = self.data[:,0]
        print(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = io.imread(img_name)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

class JPLUnlabeledTrainDataset(Dataset):
    def __init__(self, transform = None, type = "train"):
        self.root_dir = "/datasets/msl-labeled-data-set-v2.1/msl-unlabeled-data-set"
        self.image_names = []
        for f in os.listdir(self.root_dir):
            if f.endswith(".JPG"):
                self.image_names.append(os.join(self.root_dir, f))


    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = io.imread(img_name)
        
        if self.transform:
            image = self.transform(image)


        return image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}

def get_dataloaders():
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    labeled_dataset = JPLLabeledTrainDataset(transform, "train")
    labeled_dataloader = DataLoader(labeled_dataset, batch_size = 4, shuffle = True, num_workers = 0)
    unlabeled_dataset = JPLLabeledTrainDataset(transform, "train")
    unlabeled_dataloader = DataLoader(unlabeled_dataset)
    
    return labeled_dataloader, unlabeled_dataloader




labeled, unlabeled = get_dataloaders()
