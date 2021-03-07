import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import datasets, transforms
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

class OmniglotTask(Task):

    def loader(self, train: bool, batch_size):
        """
        train: true if for train dataset, false if for test dataset
        """
        
        data = torchvision.datasets.Omniglot(
            root="/datasets/", background = train, download=True, transform=torchvision.transforms.ToTensor()
        ) 
            
        dataloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 2)

        return dataloader

class OmniglotClass(OmniglotTask):
    def loss_fn(self):
        return torch.nn.CrossEntropyLoss

class OmniglotDenoising(OmniglotTask):
    def loss_fn(self):
        return torch.nn.MSELoss
    

if __name__=='__main__':
    inet_denoising = ImageNetDenoising()
    loader = inet_denoising.loader(train=True, batch_size=32)
    for images, labels in loader:
        print(images.shape, labels.shape)

class MNISTLoader():
    def get_mnist_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                ])

        data_set = datasets.MNIST('/datasets/mnist', download=True, train=True, transform=transform)
        lengths = [int(len(data_set)*0.8), int(len(data_set)*0.2)]
        train_set, val_set = random_split(train_set, lengths)

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()

        return (train_dataloader, validation_dataloader, loss_fn)

# if __name__=='__main__':
#     print('MNIST Loader: Testing')
#     test_train_dataloader, test_val_dataloader, loss_fn = MNISTLoader.get_mnist_dataloader(32)
#     inp, label = next(iter(test_train_dataloader))
#     print(inp, label)
#     inp, label = next(iter(test_val_dataloader))
#     print(inp, label)
#     print(loss_fn)
