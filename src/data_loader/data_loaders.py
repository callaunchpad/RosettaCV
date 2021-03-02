import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

class Task:
    
    def loader(self, *args):
        """
        Get Pytorch DataLoader corresponding to task data.
        """
        raise NotImplemented

    def loss_fn(self):
        raise NotImplemented

class ImageNetTask(Task):

    # override this for tasks involving pre-processing
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def loader(self, train: bool, *args):
        """
        train: whether to get loader for train or test dataset
        args: arguments for DataLoader
        NOTE: relies on installation on CSUA server
        """
        if train:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/ilsvrc2012/train/',
                                                           transform=self.transform), shuffle=True, *args)
        else:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/ilsvrc2012/val/',
                                                           transform=self.transform), shuffle=True, *args)

class ImageNetClass(ImageNetTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss

class ImageNetDenoising(ImageNetTask):
    def loss_fn(self):
        return torch.nn.MSELoss


    