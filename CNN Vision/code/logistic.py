import numpy as np

from layers import *

np.random.seed(2)

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    if hidden_dim is not None:
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim,hidden_dim))
        self.params['b1'] = np.zeros((1,hidden_dim)) 
        self.params['W2'] = np.random.normal(0, weight_scale, (1,hidden_dim))
        self.params['b2'] = np.zeros((1, )) 
    else:
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim,1))
        self.params['b1'] = 0
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the logit for X[i]
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    if 'W2' in self.params:
        # z1 = np.matmul(X,self.params['W1']) + np.matmul(np.ones((np.shape(X)[0],1)), self.params['b1'])
        z1, _ = fc_forward(X, self.params['W1'], self.params['b1'])
        a1_relu, relu_cache = relu_forward(z1)
        z2, _ = fc_forward(a1_relu, np.transpose(self.params['W2']), self.params['b2'])
        scores = z2
    else:
        z1 = np.matmul(X, self.params['W1']) + self.params['b1'] 
        scores = z1    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores.reshape(-1)
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    loss, dscores = logistic_loss(scores, y)
    if 'W2' in self.params:
        loss += 0.5 * self.reg * (np.linalg.norm(self.params['W1'])**2 + np.linalg.norm(self.params['W2'])**2)
        grads['W2'] = np.dot(np.transpose(dscores), z1) + self.reg * self.params['W2']
        grads['b2'] = np.sum(dscores)
        relu_backward_out = relu_backward(np.dot(dscores, self.params['W2']), relu_cache)
        grads['W1'] = np.dot(X.T, relu_backward_out) + self.reg * self.params['W1']
        grads['b1'] = np.dot(np.ones((1, len(y))), relu_backward_out) 
    else:
        loss += 0.5 * self.reg * (np.linalg.norm(self.params['W1'])**2)
        grads['W1'] = np.dot(np.transpose(X), dscores) + self.reg * self.params['W1']
        grads['b1'] = np.sum(dscores)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads