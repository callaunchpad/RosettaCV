import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.models as torch_models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from models.DenoisingAE import EncoderSm, DecoderSm, EncoderMd, DecoderMd, EncoderLg, DecoderLg, DenoisingAE
from models.ResNet import resnet34, resnet50
from data_loader.data_loaders import FashionMnistDenoising, ImageNetTask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7)

print('[*] Working on ' + device)

mode = 'cifar10'

if mode == 'denoisingae':
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
elif mode == 'imagenet':
    image_net_task = ImageNetTask()
    val_dl = image_net_task.loader(False, batch_size=32)

    student_model = resnet34(in_channels=3, n_classes=1000).to(device)
    
    # load state dict and convert from multigpu to generic
    checkpoint = torch.load('./trained_models/imagenet/mpl/v3-checkpoint-3-04-24.pt')
    from collections import OrderedDict
    student_model_state_dict = OrderedDict()
    for k, v in checkpoint['student_model'].items():
        name = k[7:] # remove `module.`
        student_model_state_dict[name] = v
    student_model.load_state_dict(student_model_state_dict)
    student_model.eval()

    num_batches = 0
    total_num_batches = len(val_dl)
    num_correct = 0
    num_attempted = 0

    print('[*] Testing on ' + str(total_num_batches) + ' batches of 32')

    with torch.no_grad():
        for image, label in val_dl:
            image = image / 255
            image = image.view(-1, 3, 224, 224).to(device)
            label = label.to(device)

            outputs = student_model(image)
            _, preds = torch.max(outputs, dim=-1)

            for i in range(len(label)):
                num_attempted += 1
                if label[i] == preds[i]:
                    num_correct += 1

            num_batches += 1
            if num_batches % 100 == 0:
                print('[*] Batch {}/{}'.format(num_batches, total_num_batches))

    print('[*] Test accuracy: ' + str(num_correct/num_attempted*100) + '%')

elif mode == 'cifar10':
    cifar10_dataset = CIFAR10(root="/datasets", download=True, train=False, transform=transforms.ToTensor())

    val_dl = torch.utils.data.DataLoader(cifar10_dataset, shuffle=True, batch_size=32)

    student_model = resnet50(in_channels=3, n_classes=10).to(device)
    #student_model = torch_models.resnet34(pretrained=False).to(device)
    #student_model.fc = nn.Linear(512, 10).to(device)
    
    # load state dict and convert from multigpu to generic
    checkpoint = torch.load('./trained_models/cifar10/resnet/v1-finetunempl-checkpoint-10-04-25.pt')
    from collections import OrderedDict
    student_model_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        student_model_state_dict[name] = v
    student_model.load_state_dict(student_model_state_dict)
    student_model.eval()

    num_batches = 0
    total_num_batches = len(val_dl)
    num_correct = 0
    num_attempted = 0

    print('[*] Testing on ' + str(total_num_batches) + ' batches of 32')

    with torch.no_grad():
        for image, label in val_dl:
            image = image / 255
            image = image.view(-1, 3, 32, 32).to(device)
            label = label.to(device)

            outputs = student_model(image)
            _, preds = torch.max(outputs, dim=-1)

            for i in range(len(label)):
                num_attempted += 1
                if label[i] == preds[i]:
                    num_correct += 1

            num_batches += 1
            if num_batches % 100 == 0:
                print('[*] Batch {}/{}'.format(num_batches, total_num_batches))

    print('[*] Test accuracy: ' + str(num_correct/num_attempted*100) + '%')