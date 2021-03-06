import torch
import torchvision

class OmniglotLoader():
# class OmniglotLoader(Task):
    
    def loader(self, train: bool, batch_size):
        """
        train: true if for train dataset, false if for test dataset
        """
        
        data = torchvision.datasets.Omniglot(
            root="/datasets/", background = train, download=True, transform=torchvision.transforms.ToTensor()
        ) #TODO: should automatically create folders for train and valid, change if this doesnt happen
            
        dataloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 2)

        

        return dataloader
    
    def loss_fn(self):
        return torch.nn.MSELoss #take that joey

