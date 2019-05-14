from builtins import range
import numpy as np
import operator
from scipy import signal

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
    N, Din = x.shape[0], int(np.prod(x.shape[1:]))
    x_new = np.reshape(x, (N, Din))
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
    N, D = x.shape[0], np.prod(x.shape[1:])
    x_new = np.reshape(x, (N, D))

    dx = np.dot(dout, w.T) 
    dw = np.dot(x_new.T, dout) 
    db = np.dot(dout.T, np.ones(N)) 

    dx = np.reshape(dx, x.shape)
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
    out = x * ((x > 0).astype(int))
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
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

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
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      N, D = x.shape

      # compute per-dimension mean and std_deviation
      mean = 1./N * np.sum(x, axis = 0)
      xmu = x - mean
      var = np.sqrt(1./N * np.sum(xmu ** 2, axis = 0) + eps)
      y = xmu/var
      out = gamma * y + beta
      cache = mean, xmu, var, y, gamma
      bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * mean
      bn_param['running_var'] = momentum * running_var + (1 - momentum) * var
      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      running_mean = bn_param['running_mean']
      running_var = bn_param['running_var']
      out = (gamma * ((x - running_mean) / running_var)) + beta
      #######################################################################
      #                          END OF YOUR CODE                           #
      #######################################################################
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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    N, D = dout.shape
    mean, xmu, var, y, gamma = cache
    dy = dout * gamma

    dx_num =  (N * dy - np.sum(dy, axis=0) - y*np.sum(dy*y, axis=0))
    dx_den = N* var
    dx = dx_num/dx_den 
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(y * dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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
        # mask = (np.random.rand(np.shape(x)[0], np.shape(x)[1] if len(np.shape(x)) > 1 else 1) < p).astype(int)
        # out = np.multiply(x.reshape(np.shape(mask)), mask)
        mask = (np.random.rand(*x.shape) < p)
        out = x * mask
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
    w_flipped = np.flip(np.flip(w, axis=2), axis=3)

    F, C, HH, WW = w_flipped.shape
    N, C, H, W = x.shape

    out = np.zeros((N, F, H - HH + 1, W - WW + 1))
    for im_num in range(N):
      for filt in range(F):
        for i in range(H - HH + 1):
            for j in range(W - WW + 1):
                im_part = x[im_num, :, i: i + HH, j: j + WW]
                out[im_num, filt, i, j] = np.sum(np.multiply(im_part, w[filt, :, :, :]))
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    _, _, H1, W1 = dout.shape

    dx = np.zeros((N, C, H, W))
    dw = np.zeros((F, C, HH, WW))

    for n in range(N):
      for c in range(C):
        dx_nc = np.zeros((1, 1, H, W))
        for f in range(F):
          w_field = w[f, c, :, :]
          w_field = np.expand_dims(np.expand_dims(np.pad(w_field, H1-1, 'constant'), axis=0), axis=0)
          w_flipped = np.flip(np.flip(w_field, axis=2), axis=3)
          dout_flipped = np.expand_dims(np.expand_dims(dout[n, f, :, :], axis=0), axis=0)
          dx_temp, _ = conv_forward(w_flipped, dout_flipped)
          dx_nc += np.flip(np.flip(dx_temp, axis=2), axis=3)
          # dx_nc += dx_temp
        dx[n, c, :, :] = dx_nc

      for f in range(F):
        for c in range(C):
          dw_fc = np.zeros((1, 1, HH, WW))
          for n in range(N):
            dout_field = dout[n, f, :, :]
            dout_flipped = np.expand_dims(np.expand_dims(dout_field, axis=0), axis=0)
            x_field = (np.expand_dims(np.expand_dims(x[n, c, :, :],axis=0),axis=0))
            dw_temp, _ = conv_forward(x_field, dout_flipped)
            dw_fc += dw_temp
          dw[f, c, :, :] = dw_fc

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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

    p_h = pool_param['pool_height']
    p_w = pool_param['pool_width']
    s = pool_param['stride']
    H1 = 1 + (H - p_h) // s
    W1 = 1 + (W - p_w) // s
    locs = {}

    out = np.zeros((N, C, H1, W1))

    for n in range(N):
      for h in range(H1):
        for w in range(W1):
          h_st = h * s
          h_end = h * s + p_h
          w_st = w * s
          w_end = w * s + p_w
          field = x[n, :, h_st : h_end, w_st : w_end]
          out[n, :, h, w] = np.max(field.reshape((C, p_h * p_w)), axis=1)
          location = np.argmax(field.reshape((C, p_h * p_w)), axis=1)
          for xx in range(C):
            locs[(n, xx, h, w)] = tuple(map(operator.add,np.unravel_index(location[xx],  (p_h, p_w)), (h * s, w * s)))					
         
    pool_param['locations'] = locs
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
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H1, W1 = dout.shape
    dx = np.zeros_like(x)
    for key in pool_param['locations'].keys():
      val = pool_param['locations'][key]
      grad = dout[key]
      dx[key[0], key[1], val[0], val[1]] += grad

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
    loss = np.maximum(0, 1 - y * x)
    loss = np.mean(loss)
    dx = np.zeros((len(x), 1))
    for i in range(len(x)):
        if x[i] * y[i] < 1: 
            dx[i] = -y[i]
    dx = dx/np.shape(x)[0]
    dx = dx.reshape(-1)
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
    N = y.shape[0]
    p = 1 / (1 + np.exp(-x))
    y = np.reshape(y, (np.shape(p)))
    loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    dx = (p - y)/N
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
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(x.shape[0]), y]))
    dx = probs
    dx[np.arange(x.shape[0]), y] -= 1
    dx /= x.shape[0]
    return loss, dx
