import torch
import torch.nn as nn
import torch.nn.functional as F

# Main structure of autoencoder taken from here https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html

# Small encoder and decoder
# Encoder which will embed the image to latent representation
class EncoderSm(nn.Module):
    def __init__(self, dropout=0.2):
        super(EncoderSm, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.drop2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(16, 32, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.conv3(x)

        return x

# Decoder which will upscale the embedding
class DecoderSm(nn.Module):
    def __init__(self, dropout=0.2):
        super(DecoderSm, self).__init__()

        self.convt1 = nn.ConvTranspose2d(32, 16, 5)
        self.drop1 = nn.Dropout(dropout)
        self.convt2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.drop2 = nn.Dropout(dropout)
        self.convt3 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.convt1(x))
        x = self.drop1(x)
        x = F.relu(self.convt2(x))
        x = self.drop2(x)
        x = self.convt3(x)
        x = torch.sigmoid(x)

        return x

# Medium encoder and decoder
# Encoder which will embed the image to latent representation
class EncoderMd(nn.Module):
    def __init__(self, dropout=0.2):
        super(EncoderMd, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.drop2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(32, 64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.conv3(x)

        return x

# Decoder which will upscale the embedding
class DecoderMd(nn.Module):
    def __init__(self, dropout=0.2):
        super(DecoderMd, self).__init__()

        self.convt1 = nn.ConvTranspose2d(64, 32, 5)
        self.drop1 = nn.Dropout(dropout)
        self.convt2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.drop2 = nn.Dropout(dropout)
        self.convt3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.convt1(x))
        x = self.drop1(x)
        x = F.relu(self.convt2(x))
        x = self.drop2(x)
        x = self.convt3(x)
        x = torch.sigmoid(x)

        return x

# Large encoder and decoder
# Encoder which will embed the image to latent representation
class EncoderLg(nn.Module):
    def __init__(self, dropout=0.2):
        super(EncoderLg, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.drop2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(64, 128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.conv3(x)

        return x

# Decoder which will upscale the embedding
class DecoderLg(nn.Module):
    def __init__(self, dropout=0.2):
        super(DecoderLg, self).__init__()

        self.convt1 = nn.ConvTranspose2d(128, 64, 5)
        self.drop1 = nn.Dropout(dropout)
        self.convt2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.drop2 = nn.Dropout(dropout)
        self.convt3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.convt1(x))
        x = self.drop1(x)
        x = F.relu(self.convt2(x))
        x = self.drop2(x)
        x = self.convt3(x)
        x = torch.sigmoid(x)

        return x

# Use both Encoder and Decoder structures in full Autoencoder
class DenoisingAE(nn.Module):
    def __init__(self, encoder, decoder, dropout=0.2):
        super(DenoisingAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x