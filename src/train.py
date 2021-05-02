import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.models as torch_models
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from copy import copy
from datetime import datetime

import matplotlib.pyplot as plt

from models.DenoisingAE import EncoderSm, DecoderSm, EncoderMd, DecoderMd, EncoderLg, DecoderLg, DenoisingAE
from models.ResNet import resnet50, resnet34
from models.CNN import CNN
from data_loader.data_loaders import FashionMnistDenoising, ImageNetDenoising, FashionMnistDataset
from data_loader.few_shot_dataloaders import get_few_shot_dataloader
from trainer.mpl_trainer import train_mpl
from trainer.trainer import Trainer
from utils.augmentations import *
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7)

print('[*] Training on ' + device)

mode = 'mpl'

BATCH_SIZE = 256
DATASET = 'cifar'
N_EPOCHS = 3
LR = 1e-4
WEIGHT_U = 1.5
UDA_THRESHOLD = 0.6
N_STUDENT_STEPS = 1
STOCH_DEPTH_P = 0.1
SAVE_MODEL=True

def train_cifar(model, num_epochs=50, batch_size=32, version='v1', save_model=False, optimizer=None):
    cifar10_dataset = CIFAR10(root="/datasets", download=True, transform=transforms.ToTensor())
    lengths = [int(len(cifar10_dataset)*0.6), len(cifar10_dataset) - int(len(cifar10_dataset)*0.6)]
    split_train_dataset, split_val_dataset = random_split(cifar10_dataset, lengths)
    train_dl = DataLoader(dataset=split_train_dataset, batch_size=batch_size, shuffle=True)

    model.train()

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(num_epochs):
        batch_step = 0
        total_batches = len(train_dl)
        for image, label in train_dl:
            image = image / 255

            image = image.view(-1, 3, 32, 32).to(device)
            label = label.type(torch.LongTensor).to(device)

            out_logits = model(image)
            loss = F.cross_entropy(out_logits, label)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            batch_step += 1
            if batch_step % 100 == 0:
                print('Epoch:{} Batch:{}/{} Model Loss:{:.4f}'.format(epoch+1, batch_step, total_batches, loss.item()))
    if save_model:
        checkpoint = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, 'trained_models/cifar10/resnet/' + version + '-checkpoint-' + str(num_epochs) + '-' + datetime.now().strftime('%m-%d') + '.pt')

def train_denoisingae(model, model_size, dataset, num_epochs=10, batch_size=32, learning_rate=1e-3, random_noise=0.15, save_model=False):
    # setup wandb config
    config = wandb.config
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.random_noise = random_noise
    config.model_size = model_size

    # begin training
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
    '''
    print('[*] Training MPL on FashionMNIST')
    batch_size = 32
    fashion_mnist_denoise = FashionMnistDenoising()
    train_dl = fashion_mnist_denoise.loader(True, batch_size=batch_size)
    val_dl = fashion_mnist_denoise.loader(True, batch_size=batch_size)
    '''

    '''
    # Use only train set and split into train / val
    print('[*] Training MPL on ImageNet')
    batch_size = 256

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    imageNet_dataset = ImageFolder('/datasets/imagenetwhole/ilsvrc2012/train/', transform=transform)
    lengths = [int(len(imageNet_dataset)*0.6), len(imageNet_dataset) - int(len(imageNet_dataset)*0.6)]
    split_train_dataset, split_val_dataset = random_split(imageNet_dataset, lengths)
    train_dl = DataLoader(dataset=split_train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset=split_val_dataset, batch_size=batch_size, shuffle=True)
    '''

    print('[*] Training MPL on Cifar10')
    batch_size = 32

    cifar10_dataset = CIFAR10(root="/datasets", download=True)
    lengths = [int(len(cifar10_dataset)*0.6), len(cifar10_dataset) - int(len(cifar10_dataset)*0.6)]
    split_train_dataset, split_val_dataset = random_split(cifar10_dataset, lengths)
    split_train_dataset.dataset = copy(cifar10_dataset)
    split_train_dataset.dataset.transform = TransformMPL()
    split_val_dataset.dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    train_dl = DataLoader(dataset=split_train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset=split_val_dataset, batch_size=batch_size, shuffle=True)

<<<<<<< HEAD
    # pdb.set_trace()

=======
>>>>>>> 63f895059c0876c323163f7a95ade29c2487522a
    # checkpoint = torch.load('./trained_models/cifar10/mpl/v3-checkpoint-25-04-25.pt')
    
    # train teacher model from scratch
    teacher_model = resnet34(in_channels=3, n_classes=100).to(device)
    teacher_model = nn.DataParallel(teacher_model, device_ids=[0, 1])
    #teacher_model.load_state_dict(checkpoint['teacher_model'])
    
    # train teacher model from pretrained resnet
    teacher_model = torch_models.resnet34(pretrained=True)
    teacher_model.fc = nn.Linear(512, 10)
    # freeze earlier layers (first 7 out of 10)
    ct = 0
    for child in teacher_model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    teacher_model = teacher_model.to(device)
    teacher_model = nn.DataParallel(teacher_model, device_ids=[0, 1])
    # teacher_model.load_state_dict(checkpoint['teacher_model'])
    teacher_model.load_state_dict(checkpoint['teacher_model'])

    student_model = resnet50(in_channels=3, n_classes=10).to(device)
    student_model = nn.DataParallel(student_model, device_ids=[0, 1])
    # student_model.load_state_dict(checkpoint['student_model'])

    t_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-4, weight_decay=1e-5)
    # t_optimizer.load_state_dict(checkpoint['t_optimizer'])
    s_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, weight_decay=1e-5)
    # s_optimizer.load_state_dict(checkpoint['s_optimizer'])

    with wandb.init(project="MPL-Cifar10"):
        train_mpl(teacher_model,
                        student_model,
                        train_dl,
                        val_dl,
                        batch_size=BATCH_SIZE,
                        dataset=DATASET,
                        num_epochs=N_EPOCHS,
                        learning_rate=LR,
                        weight_u=WEIGHT_U,
                        uda_threshold=UDA_THRESHOLD,
                        n_student_steps=N_STUDENT_STEPS,
                        save_model=SAVE_MODEL)
elif mode == 'baseline':
    print('[*] Training ResNet50 on Cifar10')

    model = torch_models.resnet50(pretrained=True)
    model.fc = nn.Linear(512, 10)
    # freeze earlier layers (first 7 out of 10)
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])

    train_cifar(model, num_epochs=25, batch_size=32, version='r50', save_model=True)

elif mode == 'finetune':
    print('[*] Finetuning MPL student on Cifar10')
    checkpoint = torch.load('./trained_models/cifar10/mpl/v3-checkpoint-25-04-25.pt')

    student_model = resnet50(in_channels=3, n_classes=10).to(device)
    student_model = nn.DataParallel(student_model, device_ids=[0, 1])
    student_model.load_state_dict(checkpoint['student_model'])

    s_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5, weight_decay=1e-5)
    s_optimizer.load_state_dict(checkpoint['s_optimizer'])

    train_cifar(student_model, num_epochs=10, batch_size=32, version='v1-finetunempl', save_model=True, optimizer=s_optimizer)