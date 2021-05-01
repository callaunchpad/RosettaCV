import torch.nn as nn
import torchvision.models as models


class ResNetEncoder:
    def __init__(self, device ='cpu', latent_dim=1000):
        super(ResNetEncoder, self).__init__()
        
        self.resnet18 = models.resnet18(pretrained=True)
        modules = list(self.resnet18.children())[:-1]
        modules += [nn.Linear(1000, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)] # TODO FIX BUG VERY BAD
        
        self.resnet18 = nn.Sequential(*modules).to(device)

        for p in self.resnet18.parameters():
            p.requires_grad = True

    def forward(self, x):
        # print(1, x.size())
        x = self.resnet18(x)
        # print(2, x.size())
        return x