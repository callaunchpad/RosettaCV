import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

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
        if train:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/ilsvrc2012/train/',
                                                           transform=self.transform), shuffle=True, **kwargs)
        else:
            return torch.utils.data.DataLoader(ImageFolder('/datasets/imagenetwhole/ilsvrc2012/val/',
                                                           transform=self.transform), shuffle=True, **kwargs)

class ImageNetClass(ImageNetTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss

class ImageNetDenoising(ImageNetTask):
    def loss_fn(self):
        return torch.nn.MSELoss

if __name__=='__main__':
    inet_denoising = ImageNetDenoising()
    loader = inet_denoising.loader(train=True, batch_size=32)
    for images, labels in loader:
        print(images.shape, labels.shape)
