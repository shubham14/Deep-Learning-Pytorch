from builtins import range
import numpy as np
from utils import *


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    N, D = x.shape[0], np.prod(x.shape[1:])
    x_new = np.reshape(x, (N, D))
    out = np.dot(x_new, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x, (N, D))

    dx2 = np.dot(dout, w.T) # N x D
    dw = np.dot(x2.T, dout) # D x M
    db = np.dot(dout.T, np.ones(N)) # M x 1

    dx = np.reshape(dx2, x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x
    out[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.multiply((x > 0).astype(int), dout) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #######################################################################

        # Forward pass
        # Step 1 - shape of mu (D,)
        mu = 1 / float(N) * np.sum(x, axis=0)

        # Step 2 - shape of var (N,D)
        xmu = x - mu

        # Step 3 - shape of carre (N,D)
        carre = xmu**2

        # Step 4 - shape of var (D,)
        var = 1 / float(N) * np.sum(carre, axis=0)

        # Step 5 - Shape sqrtvar (D,)
        sqrtvar = np.sqrt(var + eps)

        # Step 6 - Shape invvar (D,)
        invvar = 1. / sqrtvar

        # Step 7 - Shape va2 (N,D)
        va2 = xmu * invvar

        # Step 8 - Shape va3 (N,D)
        va3 = gamma * va2

        # Step 9 - Shape out (N,D)
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, carre, var, sqrtvar, invvar,
                 va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #######################################################################
        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    ##########################################################################
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    # Backprop Step 9
    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    # Backprop step 8
    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)

    # Backprop step 7
    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)

    # Backprop step 6
    dsqrtvar = -1. / (sqrtvar**2) * dinvvar

    # Backprop step 5
    dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar

    # Backprop step 4
    dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar

    # Backprop step 3
    dxmu += 2 * xmu * dcarre

    # Backprop step 2
    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)

    # Basckprop step 1
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) > dropout_param['p']
        out = np.multiply(x, mask)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = None
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dout *= mask
        dx = dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad = 0
    stride = 1

    H_prime = int(1 + (H - HH))
    W_prime = int(1 + (W - WW))
    out = np.zeros((N, F, H_prime, W_prime))
    for i in range(N):
        w_flipped = np.flip(w)
        for j in range(H_prime):
            for k in range(W_prime):
                h_start = j * stride
                h_end = h_start + HH

                w_start = k * stride
                w_end = w_start + WW
                
                #TODO: get local receptive field of shape C x HH x WW
                x_field = x[:, :, h_start:h_end, w_start:w_end] 

                #TODO: compute the features for each convolutional filter
                for f in range(F):
                    out[i, f, j, k] = np.sum(x_field * w_flipped[f]) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


# def conv_backward(dout, cache):
#     """
#     A naive implementation of the backward pass for a convolutional layer.
#     Inputs:
#     - dout: Upstream derivatives.
#     - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
#     Returns a tuple of:
#     - dx: Gradient with respect to x
#     - dw: Gradient with respect to w
#     - db: Gradient with respect to b
#     """
#     dx, dw = None, None
#     x, w = cache
#     #############################################################################
#     # TODO: Implement the convolutional backward pass.                          #
#     #############################################################################
#     N, C, H, W = x.shape
#     F, _, HH, WW = w.shape
#     _, _, H_out, W_out = dout.shape
#     stride, pad = 1, 0

#     dx = np.zeros_like(x)
#     dw = np.zeros_like(w)


#     # Ref: https://github.com/MahanFathi/CS231/blob/ecab92ed8627ea0ea513a54fc2019516d446c106/assignment2/cs231n/layers.py#L483-L491
#     for n in range(N):
#     	for f in range(F):
#     		for h_out in range(H_out):
#     			for w_out in range(W_out):
#     				dw[f] += x[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] * dout[n, f, h_out, w_out]
#     				dx[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] += w[f] * dout[n, f, h_out, w_out]

#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx, dw


def conv_backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w= cache

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_prime, W_prime = dout.shape

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    stride = 1
    for n in range(N):
        for j in range(H_prime):
            for k in range(W_prime):
                h_start = j * stride
                h_end = h_start + HH
                w_start = k * stride
                w_end = w_start + WW

                local_receptive_field = x[:, :, h_start:h_end, w_start:w_end]  
                
                for c in range(C):
                    for f in range(F):
                        dw[f, c, :, :] += convolutionValid2d(dout[n, f, :, :], local_receptive_field[n, c, :, :])
                        dx[n, c, :, :] += convolutionFull2d(w[f, c, :, :], dout[n, f, :, :])
                        
    return dx, dw

def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    h_p = pool_param['pool_height']
    w_p = pool_param['pool_width']
    s = pool_param['stride']
    H_prime = 1 + (H - h_p) // s
    W_prime = 1 + (W - w_p) // s

    out = np.zeros((N, C, H_prime, W_prime))

    for n in range(N):
        for h in range(H_prime):
            for w in range(W_prime):
                h1 = h * s
                h2 = h * s + h_p
                w1 = w * s
                w2 = w * s + w_p
                max_pool_field = x[n, :, h1:h2, w1:w2]
                out[n,:,h,w] = np.max(max_pool_field.reshape((C, h_p * w_p)), axis=1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache 
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    h_p = pool_param['pool_height']
    w_p = pool_param['pool_width']
    s = pool_param['stride']
    H_prime = 1 + (H - h_p) // s
    W_prime = 1 + (W - w_p) // s

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(H_prime):
                for w in range(W_prime):
                    h1 = h * s
                    h2 = h * s + h_p
                    w1 = w * s
                    w2 = w * s + w_p
                    window = x[n, c, h1:h2, w1:w2]
                    window2 = np.reshape(window, (h_p * w_p))
                    window3 = np.zeros_like(window2)
                    window3[np.argmax(window2)] = 1

                    dx[n, c, h1:h2, w1:w2] = np.reshape(window, (h_p, w_p)) * dout[n,c,h,w]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient for binary SVM classification.
    Inputs:
    - x: Input data, of shape (N,) where x[i] is the score for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    x = x.reshape(-1)
    y = 2 * y - 1
    loss = np.zeros((len(x), 1))
    for i in range(len(x)):
        loss[i] = max([0, 1 - x[i] * y[i]])
    loss = np.mean(loss)
    dx = np.zeros((len(x), 1))
    for i in range(len(x)):
        if x[i] * y[i] < 1: 
            dx[i] = -y[i]
    dx = dx/np.shape(x)[0]
    return loss, dx


def logistic_loss(x, y):
    """
    Computes the loss and gradient for binary classification with logistic 
    regression.
    Inputs:
    - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # x = x.reshape(-1)
    N = y.shape[0]
    p = 1 / (1 + np.exp(-x))
    y = np.reshape(y, (np.shape(p)))
    loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    dx = (p - y)/N
    # dx = dx.reshape(-1)
    print(dx.shape)
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x)
    probs /= np.sum(probs)
    N = x.shape[0]
    loss = -np.mean(np.log(probs[np.arange(N), y]))
    dx = probs
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
