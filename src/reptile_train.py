import wandb
import argparse

import torch
import torchvision.models as models
import torch.nn as nn

from trainer.meta_trainer import meta_outer_train_loop, reptile_inner_train_loop, reptile_update_params
from data_loader.data_loaders import OmniglotByAlphabet

def change_resnet_labels(num_labels):
    model_ft = models.resnet18(pretrained=False)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, num_labels)
    return model_ft

def train_reptile(args):
    model = change_resnet_labels(14)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
    loader = OmniglotByAlphabet(characters_per_alphabet=14, force_constant_size=True)
    train_tasks = loader.get_train_tasks()
    validation_tasks = loader.get_validation_tasks()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_outer_train_loop(model, optimizer, train_tasks, validation_tasks, args.batch_size, args.num_iters_inner,
                     args.num_iters_outer, reptile_inner_train_loop, reptile_update_params, device, args.save_name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-in', '--num_iters_inner', default=32)
    ap.add_argument('-out', '--num_iters_outer', default=1000)
    ap.add_argument('-ilr', '--inner_lr', default=1e-3)
    ap.add_argument('-olr', '--outer_lr', default=0.2)
    ap.add_argument('--batch_size', default=10)
    ap.add_argument('--save_name', default='meta-test.p')
    args = ap.parse_args()

    wandb.init('RosettaCV')
    wandb.config.update(args)
    wandb.config.model = "resnet18"
    wandb.config.type = "reptile"
    train_reptile(args)
