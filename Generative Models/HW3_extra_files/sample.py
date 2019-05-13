# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:11:49 2019

@author: Shubham
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

use_cuda = False
input_size = 28 * 28
units = 400
batch_size = 32
latent_size = 20 # z dim
num_classes = 10
num_epochs = 11

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
        
