import numpy as np
import numpy.linalg as LA
from layers import *

class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.hidden_dim = hidden_dim
    self.filter_size = filter_size
    self.reg = reg
    self.input_dim = input_dim
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim

    print('Using Xavier Intialization')
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
    self.params['W2'] = np.random.randn((input_dim[1] - filter_size + 1) // 2 * (input_dim[2]- filter_size + 1) // 2 * num_filters,hidden_dim)*np.sqrt(2/( ((input_dim[1] - filter_size + 1) // 2 * (input_dim[2]- filter_size + 1) // 2 * num_filters+hidden_dim)))
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * np.sqrt(2/(hidden_dim+num_classes))
    self.params['b2'] = np.zeros((1,hidden_dim))
    self.params['b3'] = np.zeros((1,num_classes))
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    
    conv_param = {'stride': 1, 'pad': 0}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    if y is None:
        dropout_param = {'p':0.8, 'mode':'test', 'seed':42}
    else:
        dropout_param = {'p':0.8, 'mode':'train', 'seed':42}

    running_mean = np.zeros((1, self.hidden_dim))
    running_var = np.zeros((1, self.hidden_dim))
    bn_param = {'mode':'train', 'eps':1e-8, 'momentum':0.9, }


    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_cache = conv_forward(X, W1)
    relu_out, relu_cache = relu_forward(conv_out)
    max_pool_out, max_pool_cache = max_pool_forward(relu_out, pool_param)
    D = np.prod(max_pool_out.shape[1:])
    max_pool_out1 = np.reshape(max_pool_out, (-1, D))
    affine_output_1, affine_cache_1 = fc_forward(max_pool_out1, W2, b2)
    affine_output_2, affine_cache_2 = fc_forward(affine_output_1, W3, b3)
    scores = affine_output_2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    C, H, W = self.input_dim
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * ((np.linalg.norm(self.params['W1']))**2 + (np.linalg.norm(self.params['W2']))**2 + (np.linalg.norm(self.params['W3']))**2)
    dx_affine3, dw3, db3 = fc_backward(dscores, affine_cache_2)
    dx_affine2, dw2, db2 = fc_backward(dx_affine3, affine_cache_1)
    dx_affine2 = np.reshape(dx_affine2, max_pool_out.shape)
    d_maxpool = max_pool_backward(dx_affine2, max_pool_cache)
    d_relu = relu_backward(d_maxpool, relu_cache)
    dx_conv, dw1 = conv_backward(d_relu, conv_cache)

    # saving the parameters
    grads['W1'] = dw1 + self.reg*W1
    grads['W2'] = dw2 + self.reg*W2
    grads['b2'] = db2 
    grads['W3'] = dw3 + self.reg*W3
    grads['b3'] = db3 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads