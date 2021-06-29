import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import transforms
import numpy as np
import torch.utils.data as utils
import torch.nn as nn
import torchvision

class GRL(Function):
    """
    The gradient reversal layer acts as an identity transformation during forward propogation.
    However, during backprop, it takes the gradient and changes its sign (i.e it multiplies it by -1).
    """
    @staticmethod
    def forward(ctx, x, constant):
        """
        "ctx" is a context object.
        """
        net.constant = constant
        return x.view_as(x) * constant
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

class Dann(nn.Module):
    """
    Implementation of the Domain Adversarial Neural Network, used for implementing Domain Confusion loss.
    """
    def __init__(self, latent_dim, num_domains):
        super(Dann, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 100)
        self.batchnorm = nn.batchnorm(100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, num_domains)
    
    def forward(self, x, alpha = 1):
        """
        alpha is a constant by which the gradient can be multiplied.
        Leaving it at 1 basically means we simply negate it.
        """
        x = GRL.apply(x)
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(x.shape[0], -1)
        return x


