import torch
import torch.nn as nn
import wandb

from data_loader.data_loaders import FashionMnistDenoising
device = 'cuda' if torch.cuda.is_available() else 'cpu'

WANDB = False

def train(model,
          task,
          num_epochs: int=10,
          batch_size: int=32,
          learning_rate: float=1e-3,
          noise_ratio: float=0.15,
          seed=None):
    print("[*] Training DenoisingAE")
    if seed:
        torch.seed(seed)
    criterion = task.loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_dl = task.loader(train=True, batch_size=32)

    for epoch in range(num_epochs):
        for img, _ in train_dl:
            img = img.to(device)
            img = img / 255
            img = img.view(-1, 28, 28)
            mask = (torch.rand(*img.shape) > noise_ratio).to(device)
            noisy_imgs = (img*mask).to(device).unsqueeze(1)

            out = model(noisy_imgs)
            img = img.unsqueeze(1)
            loss = criterion(out, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if USE_WANDB:
            wandb.log({ "epoch": epoch + 1, "loss": float(loss) })
        print('Epoch:{} Loss:{:.4f}'.format(epoch+1, float(loss)))

if __name__=="__main__":
    from models.DenoisingAE import Encoder, Decoder, DenoisingAE
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    fashion_mnist_denoise = FashionMnistDenoising()
    model = DenoisingAE(encoder, decoder).to(device)

    if USE_WANDB:
        with wandb.init(project="DenoisingAE"):
            train(model, fashion_mnist_denoise)
            wandb.alert(title="Train DenoisingAE", text="Finished training")
    else:
        train(model, fashion_mnist_denoise)