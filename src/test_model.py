import torch
import torchvision
import matplotlib.pyplot as plt

from models.DenoisingAE import EncoderSm, DecoderSm, EncoderMd, DecoderMd, EncoderLg, DecoderLg, DenoisingAE
from data_loader.data_loaders import FashionMnistDenoising

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('[*] Working on ' + device)

fashion_mnist_denoise = FashionMnistDenoising()

train_dl = fashion_mnist_denoise.loader(True, batch_size=32)

encoder = EncoderMd().to(device)
decoder = DecoderMd().to(device)

model = DenoisingAE(encoder, decoder).to(device)
model.load_state_dict(torch.load('./trained_models/fashion_mnist/denoisingae_md/denoisingae-03-14-0.3.pt'))

random_noise = 0.3

for img, _ in train_dl:
    img = img.to(device)
    img = img.view(-1, 28, 28)

    plt.imshow(img[0].cpu())
    plt.show()
    
    img = img / 255

    noisy_imgs = []
    for i in range(img.shape[0]):
        noisy_mask = torch.FloatTensor(28, 28).to(device).uniform_() > random_noise
        noisy_imgs.append(img[i] * noisy_mask)
    noisy_imgs = torch.stack(noisy_imgs)
    plt.imshow(noisy_imgs[0].cpu())
    plt.show()
    noisy_imgs = noisy_imgs.unsqueeze(1).to(device)

    out = model(noisy_imgs)
    img = img.unsqueeze(1)

    plt.imshow(img[0][0].cpu())
    plt.show()