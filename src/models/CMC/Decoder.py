import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, image_h, image_w, latent_dim=512, reshape_size=4):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.reshape_size = reshape_size

        self.l1 = nn.Linear(latent_dim, 1024)

        self.l2 = nn.Linear(1024, 2400)

        self.convt1 = nn.ConvTranspose2d(2400//(15*20), 16, 5, stride=2, padding=2, output_padding=1)
        # self.drop1 = nn.Dropout(dropout)
        self.convt2 = nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1)
        # self.drop2 = nn.Dropout(dropout)
        self.convt3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        # self.convt6 = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        x = x.view(x.shape[0], -1, 15, 20)

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))
        x = self.convt5(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    dec = Decoder()