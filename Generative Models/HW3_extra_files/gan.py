import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    ###########################
    ######### TO DO ###########
    ###########################
    random_noise = None
    random_noise = torch.randn((batch_size, dim))
    return random_noise


def build_discriminator(batch_size):
    """
    Build and return a PyTorch model for the DCGAN discriminator
    using the architecture described below:

    * Reshape into image tensor (Use Unflatten!)
    * 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
    * Max Pool 2x2, Stride 2
    * 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
    * Max Pool 2x2, Stride 2
    * Flatten
    * Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
    * Fully Connected size 1

    """
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        ###########################
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, 5),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 5),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2),
        Flatten(),
        nn.Linear(4 * 4 * 64, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 1)
    )


def build_generator(batch_size, noise_dim):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described below:

    * Fully connected of size 1024, ReLU
    * BatchNorm
    * Fully connected of size 7 x 7 x 128, ReLU
    * BatchNorm
    * Reshape into Image Tensor
    * 64 conv2d^T filters of 4x4, stride 2, 'same' padding, ReLU
    * BatchNorm
    * 1 conv2d^T filter of 4x4, stride 2, 'same' padding, TanH
    * Should have a 28x28x1 image, reshape back into 784 vector

    Note: for conv2d^T you should use torch.nn.ConvTranspose2d
          Plese see the documentation for it in the pytorch site

    """
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        ###########################
        nn.Linear(noise_dim, 1024),
        nn.ReLU(True),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(True),
        nn.BatchNorm1d(7 * 7 * 128),
        Unflatten(batch_size, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        nn.Tanh(),
        Flatten()
    )


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3, betas=(0.5, 0.999))
    return optimizer


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    neg_abs = - input.abs()
    z = torch.zeros_like(neg_abs)
    loss1 = torch.max(input, z) - input * target + (1 + neg_abs.exp()).log()
    loss = loss1.mean()
    return loss


def discriminator_loss(logits_real, logits_fake, dtype):
    """
    Computes the discriminator loss described in the homework pdf
    using the bce_loss function.
    
    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    loss = None
    dis_targets = torch.ones_like(logits_real)
    loss1 = bce_loss(logits_real, dis_targets)
    
    gen_targets = torch.zeros_like(logits_fake)
    loss2 = bce_loss(1 - logits_fake, gen_targets)
    loss = loss1 + loss2
    return loss


def generator_loss(logits_fake, dtype):
    """
    Computes the generator loss described in the homework pdf using
    the bce_loss function.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    loss = None
    dis_targets = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, dis_targets)
    return loss


def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,
              loader_train, dtype, show_every=250, batch_size=128, noise_size=96,
              num_epochs=10):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = Variable(x).type(dtype)
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake, dtype)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake, dtype)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(
                    iter_count,d_total_error.data,g_error.data))
                imgs_numpy = fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.pause(1.0)
                print()
            iter_count += 1


def main():

    NUM_TRAIN = 50000
    NUM_VAL = 5000
    
    NOISE_DIM = 96
    batch_size = 128
    
    mnist_train = dset.MNIST('./data', train=True, download=True,
                               transform=T.ToTensor())
    loader_train = DataLoader(mnist_train, batch_size=batch_size,
                              sampler=ChunkSampler(NUM_TRAIN, 0))
    
    mnist_val = dset.MNIST('./data', train=True, download=True,
                               transform=T.ToTensor())
    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
    
    
    imgs = loader_train.__iter__().next()[0].view(
        batch_size, 784).numpy().squeeze()
    show_images(imgs)

    # dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

    D_DC = build_discriminator(batch_size).type(dtype) 
    D_DC.apply(initialize_weights)
    G_DC = build_generator(batch_size, NOISE_DIM).type(dtype)
    G_DC.apply(initialize_weights)
    
    D_DC_solver = get_optimizer(D_DC)
    G_DC_solver = get_optimizer(G_DC)
    
    run_a_gan(D_DC, G_DC, D_DC_solver, G_DC_solver, discriminator_loss,
              generator_loss, loader_train, dtype, num_epochs=5)


if __name__ == "__main__":
    main()
