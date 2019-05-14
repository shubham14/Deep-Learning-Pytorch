import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import PIL

import numpy as np

from scipy.misc import imread
from collections import namedtuple
import matplotlib.pyplot as plt

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, size=512):
    transform = T.Compose([
        T.Scale(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.
    
    Inputs:
    - x: A PyTorch Variable of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.
    
    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Variable of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def content_loss(content_weight, content_current, content_target):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    """
    sub = content_current - content_target
    return content_weight * torch.sum(sub ** 2)
    


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N, C, H, W = features.shape
    G = features[..., None] * features[..., None].permute(0, 4, 2, 3, 1)
    G = G.sum(dim=[2, 3])
    if normalize:
        return G / (H * W * C)
    else:
        return G


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    L_s = 0
    for i in range(len(style_targets)):
        G_l = gram_matrix(feats[style_layers[i]])
        sub = G_l - style_targets[i]
        sub_sq_sum = torch.sum(sub ** 2)
        L_s += style_weights[i] * sub_sq_sum
    return L_s



def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    _, _, H, W = img.size()
    img_h1 = img[:, :, 0:H-1, :]
    img_h2 = img[:, :, 1:H , :]

    img_w1 = img[:, :, :, 0:W-1]
    img_w2 = img[:, :, :, 1:W]
    
    horizontal_diff = img_h1 - img_h2
    vertical_diff = img_w1 - img_w2

    diff = torch.sum(horizontal_diff ** 2) + torch.sum(vertical_diff ** 2)
    loss = tv_weight * diff
    return loss


def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
    """
    Run style transfer!
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """

    dtype = torch.FloatTensor
    # Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
#    dtype = torch.cuda.FloatTensor

    # Load the pre-trained SqueezeNet model.
    cnn = torchvision.models.squeezenet1_1(pretrained=True).features
    cnn.type(dtype)

    # We don't want to train the model any further, so we don't want PyTorch to waste computation 
    # computing gradients on parameters we're never going to update.
    for param in cnn.parameters():
        param.requires_grad = False

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))
    feats = extract_features(content_img_var, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img_var = Variable(img, requires_grad=True)

    # loss vector for plotting curves
    l = []
    it = []
    # Set up optimization hyperparameters
    initial_lr = 3
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img_var Torch variable, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img_var], lr=initial_lr)
    
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu()))
    axarr[1].imshow(deprocess(style_img.cpu()))
    plt.show()
    plt.figure()
    
    for t in range(200):
        if t < 190:
            img.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(img_var, cnn)
        
        #TODO:Compute loss
        L_c = content_loss(content_weight, feats[content_layer], content_target)
        L_s = style_loss(feats, style_layers, style_targets, style_weights)
        L_tv = tv_loss(img, tv_weight)
        loss = L_c + L_s + L_tv

        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img_var], lr=decayed_lr)
        optimizer.step()
        
        print('Iteration %d, loss %g'%(t, loss))
        l.append(loss)
        it.append(t)
        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.axis('off')
            plt.imshow(deprocess(img.cpu()))
            plt.show()
    a = plt.plot(it, l)
    plt.show()
    save_name = content_image + 'loss_plot.jpg'
    plt.savefig(save_name)
    print('Iteration {}'.format(t))
    plt.axis('off')
    plt.imshow(deprocess(img.cpu()))
    plt.show()

def main():
    # Composition VII + Tubingen
    params1 = {
        'content_image' : 'styles/tubingen.jpg',
        'style_image' : 'styles/composition_vii.jpg',
        'image_size' : 192,
        'style_size' : 512,
        'content_layer' : 3,
        'content_weight' : 5e-2, 
        'style_layers' : (1, 4, 6, 7),
        'style_weights' : (20000, 500, 12, 1),
        'tv_weight' : 5e-2
    }

    style_transfer(**params1)

    # Scream + Tubingen
    params2 = {
        'content_image':'styles/tubingen.jpg',
        'style_image':'styles/the_scream.jpg',
        'image_size':192,
        'style_size':224,
        'content_layer':3,
        'content_weight':3e-2,
        'style_layers':[1, 4, 6, 7],
        'style_weights':[200000, 800, 12, 1],
        'tv_weight':2e-2
    }

    style_transfer(**params2)

    # Starry Night + Tubingen
    params3 = {
        'content_image' : 'styles/tubingen.jpg',
        'style_image' : 'styles/starry_night.jpg',
        'image_size' : 192,
        'style_size' : 192,
        'content_layer' : 3,
        'content_weight' : 6e-2,
        'style_layers' : [1, 4, 6, 7],
        'style_weights' : [300000, 1000, 15, 3],
        'tv_weight' : 2e-2
    }

    style_transfer(**params3)

if __name__== "__main__":
    main()
