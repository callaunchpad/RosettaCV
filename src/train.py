import torch
import torch.nn as nn
import wandb
from datetime import datetime

from models.DenoisingAE import Encoder, Decoder, DenoisingAE
from data_loader.data_loaders import FashionMnistDenoising

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, num_epochs=10, batch_size=32, learning_rate=1e-3, random_noise=0.15, save_model=False):
    # setup wandb config
    config = wandb.config
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.random_noise = random_noise

    # begin training
    print("[*] Training DenoisingAE on FashionMNIST")
    fashion_mnist_denoise = FashionMnistDenoising()
    torch.manual_seed(7)
    criterion = fashion_mnist_denoise.loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_dl = fashion_mnist_denoise.loader(True, batch_size=32)

    for epoch in range(num_epochs):
        for img, _ in train_dl:
            img = img.to(device)
            img = img / 255
            img = img.view(-1, 28, 28)

            noisy_imgs = []
            for i in range(img.shape[0]):
                noisy_mask = torch.FloatTensor(28, 28).to(device).uniform_() > random_noise
                noisy_imgs.append(img[i] * noisy_mask)
            noisy_imgs = torch.stack(noisy_imgs)
            noisy_imgs = noisy_imgs.unsqueeze(1).to(device)

            out = model(noisy_imgs)
            img = img.unsqueeze(1)
            loss = criterion(out, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        wandb.log({ "epoch": epoch + 1, "loss": float(loss) })
        print('Epoch:{} Loss:{:.4f}'.format(epoch+1, float(loss)))
    
    if save_model:
        torch.save(model.state_dict(), 'trained_models/denoisingae-' + datetime.now().strftime('%m-%d') + '-' + str(random_noise) + '.pt')

r_noise = [0.10, 0.15, 0.20, 0.30, 0.40]

for noise_amt in r_noise:
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = DenoisingAE(encoder, decoder).to(device)

    with wandb.init(project="DenoisingAE"):
        train(model, random_noise=noise_amt, save_model=True)
        #wandb.alert(title="Train DenoisingAE", text="Finished training")