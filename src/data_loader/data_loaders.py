import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

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

if __name__=='__main__':
    print('MNIST Loader: Testing')
    test_train_dataloader, test_val_dataloader, loss_fn = MNISTLoader.get_mnist_dataloader(32)
    inp, label = next(iter(test_train_dataloader))
    print(inp, label)
    inp, label = next(iter(test_val_dataloader))
    print(inp, label)
    print(loss_fn)
