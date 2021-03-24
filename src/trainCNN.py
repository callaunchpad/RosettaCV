import torch.nn as nn
from torchvision.datasets import ImageFolder

from models.CNN import CNN
from trainer.trainer import Trainer
from data_loader.few_shot_dataloaders import SubSampledDataset

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

train_set = ImageFolder('/datasets/imagenetwhole/ilsvrc2012/train/', transform=self.transform)
val_set = ImageFolder('/datasets/imagenetwhole/ilsvrc2012/val/', transform=self.transform)

model = CNN()
loss = nn.MSELoss()
train_loader = SubSampledDataset.get_few_shot_dataloader(train_set)
val_loader = SubSampledDataset.get_few_shot_dataloader(val_set)

trainer = Trainer(model, loss, train_loader, val_loader)
trainer.train(epochs=100)
