import torch.nn as nn
import torchvision.models as models
import torch


class ResNetEncoder(nn.Module):
    def __init__(self, device ='cpu', latent_dim=512):
        super(ResNetEncoder, self).__init__()
        
        self.resnet18 = models.resnet18(pretrained=True)
        modules = list(self.resnet18.children())[:-1]
        
        self.resnet18 = nn.Sequential(*modules).to(device)

        self.l1 = nn.Linear(512, latent_dim)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(latent_dim, latent_dim)

        for p in self.resnet18.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.resnet18(x)
        x = torch.squeeze(x)
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x