import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, image_h, image_w, latent_dim, reshape_size=5):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.reshape_size = reshape_size

        self.convt1 = nn.ConvTranspose2d(self.latent_dim//(self.reshape_size*self.reshape_size), 64, 5)
        # self.drop1 = nn.Dropout(dropout)
        self.convt2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        # self.drop2 = nn.Dropout(dropout)
        self.convt3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):

        x = x.view(-1, self.reshape_size, self.reshape_size, self.latent_dim//25)
        x = F.relu(self.convt1(x))
        # x = self.drop1(x)
        x = F.relu(self.convt2(x))
        # x = self.drop2(x)
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))
        x = self.convt5(x)
        
        x = torch.sigmoid(x)

        print("lol3", x.shape)

        return x