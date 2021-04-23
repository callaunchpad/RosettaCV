import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import wandb
from datetime import datetime

import matplotlib.pyplot as plt

from models.DenoisingAE import EncoderSm, DecoderSm, EncoderMd, DecoderMd, EncoderLg, DecoderLg, DenoisingAE
from models.ResNet import resnet34
from models.CNN import CNN
from data_loader.data_loaders import FashionMnistDenoising, ImageNetDenoising, FashionMnistDataset
from data_loader.few_shot_dataloaders import get_few_shot_dataloader
from trainer.mpl_trainer import train_mpl
from trainer.trainer import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('[*] Training on ' + device)

mode = 'mpl'

if mode == 'denoisingae':
    #r_noise = [0.10, 0.15, 0.20, 0.30, 0.40]
    r_noise = [0.40]
    model_size = 'sm' # sm, md, or lg

    for noise_amt in r_noise:
        if model_size == 'sm':
            encoder = EncoderSm().to(device)
            decoder = DecoderSm().to(device)
        elif model_size == 'md':
            encoder = EncoderMd().to(device)
            decoder = DecoderMd().to(device)
        else:
            encoder = EncoderLg().to(device)
            decoder = DecoderLg().to(device)

        model = DenoisingAE(encoder, decoder).to(device)
        model = nn.DataParallel(model, device_ids=[0, 1])

        with wandb.init(project="DenoisingAE"):
            train_denoisingae(model, model_size, 'imagenet', num_epochs=5, random_noise=noise_amt, save_model=True)
            #wandb.alert(title="Train DenoisingAE", text="Finished training")
elif mode == 'mpl':
    print('[*] Training MPL on FashionMNIST')
    
    torch.manual_seed(7)
    batch_size = 32
    fashion_mnist_denoise = FashionMnistDenoising()
    train_dl = fashion_mnist_denoise.loader(True, batch_size=batch_size)
    val_dl = fashion_mnist_denoise.loader(True, batch_size=batch_size)

    teacher_model = resnet34(in_channels=1, n_classes=10).to(device)
    student_model = resnet34(in_channels=1, n_classes=10).to(device)

    with wandb.init(project="MPL-FashionMNIST"):
        train_mpl(teacher_model, student_model, train_dl, val_dl, batch_size, 'fashion_mnist', num_epochs=20, save_model=True)

def train_denoisingae(model, model_size, dataset, num_epochs=10, batch_size=32, learning_rate=1e-3, random_noise=0.15, save_model=False):
    # setup wandb config
    config = wandb.config
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.random_noise = random_noise
    config.model_size = model_size

    # begin training
    torch.manual_seed(7)
    if dataset == 'fashion_mnist':
        print('[*] Training DenoisingAE on FashionMNIST')
        fashion_mnist_denoise = FashionMnistDenoising()
        criterion = fashion_mnist_denoise.loss_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        train_dl = fashion_mnist_denoise.loader(True, batch_size=32)
    elif dataset == 'imagenet':
        print('[*] Training DenoisingAE on ImageNet')
        imagenet_dnoise = ImageNetDenoising()
        imagenet_dnoise.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        criterion = imagenet_dnoise.loss_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        train_dl = imagenet_dnoise.loader(True, batch_size=32)

    global_step = 0
    for epoch in range(num_epochs):
        batch_num = 0
        total_batches = len(train_dl)
        for img, _ in train_dl:
            img = img.to(device)
            img = img / 255

            '''display test image
            plt.imshow(img[0][0].cpu())
            plt.show()
            '''

            if dataset == 'fashion_mnist':
                img = img.view(-1, 28, 28) # convert to minibatches of 1
            elif dataset == 'imagenet':
                img = img.view(-1, 224, 224) # convert to minibatches of 1

            noisy_imgs = []
            for i in range(img.shape[0]):
                if dataset == 'fashion_mnist':
                    noisy_mask = torch.FloatTensor(28, 28).to(device).uniform_() > random_noise
                elif dataset == 'imagenet':
                    noisy_mask = torch.FloatTensor(224, 224).to(device).uniform_() > random_noise
                
                noisy_imgs.append(img[i] * noisy_mask)
            noisy_imgs = torch.stack(noisy_imgs)
            if dataset == 'fashion_mnist':
                noisy_imgs = noisy_imgs.view(-1, 1, 28, 28).to(device)
            elif dataset == 'imagenet':
                noisy_imgs = noisy_imgs.view(-1, 1, 224, 224).to(device)

            out = model(noisy_imgs)
            img = img.unsqueeze(1)
            loss = criterion(out, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            batch_num += 1
            if global_step % 100 == 0:
                print('Epoch:{} Batch:{}/{} Loss:{:.4f}'.format(epoch+1, batch_num, total_batches, float(loss)))
                wandb.log({ 'batch_loss': float(loss) })

        wandb.log({ 'epoch': epoch + 1, 'loss': float(loss) })
        print('Epoch:{} Loss:{:.4f}'.format(epoch+1, float(loss)))
    
    if save_model:
        torch.save(model.state_dict(), 'trained_models/' + dataset + '/denoisingae_' + model_size + '/denoisingae-' + datetime.now().strftime('%m-%d') + '-' + str(random_noise) + '.pt')