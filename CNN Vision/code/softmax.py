import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        if hidden_dim is not None:
            W1 = np.random.normal(scale=weight_scale, size=(self.input_dim, self.hidden_dim))
            b1 = np.zeros((1, self.hidden_dim))
            W2 = np.random.normal(scale=weight_scale, size=(self.hidden_dim, self.num_classes))
            b2 = np.zeros((1, self.num_classes))
            self.params['W1'] = W1
            self.params['b1'] = b1
            self.params['W2'] = W2
            self.params['b2'] = b2
        else:
            W1 = np.random.normal(scale=weight_scale, size=(self.input_dim, self.num_classes))
            b1 = np.zeros((1, self.num_classes))
            self.params['W1'] = W1
            self.params['b1'] = b1

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
        if self.hidden_dim is not None:
            z1 = np.dot(X, self.params['W1']) + self.params['b1']
            a1, _ = relu_forward(z1)
            z2 = np.dot(a1,self.params['W2']) + self.params['b2']
            scores = z2
        else:
            z1 = np.dot(X,self.params['W1']) + self.params['b1']
            scores = z1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
          return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the model. Store the loss          #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss and make sure that grads[k] holds the gradients for self.params[k]. #
        # Don't forget to add L2 regularization.                                   #
        #                                                                          #
        ############################################################################
        if self.hidden_dim is not None:
            W2 = self.params['W2']
            W1 = self.params['W1']

            loss, dscores = softmax_loss(scores, y)
            loss = loss + 0.5 * self.reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2']))) 
            dw2 = np.dot(a1.T, dscores)
            dw2 = np.reshape(dw2, (self.hidden_dim, self.num_classes))
            dw2 += self.reg * W2
            db2 = np.sum(dscores, axis=0)

            grads['W2'] = dw2
            grads['b2'] = db2

            dz1 = np.dot(dscores ,self.params['W2'].T)
            relu_mask = (a1 > 0)
            dr1 = relu_mask * dz1
            dw1 = np.dot(X.T, dz1)
            
            dw1 += self.reg * self.params['W1']
            dw1 = np.reshape(dw1, (self.input_dim, self.hidden_dim))
            grads['W1'] = dw1 
            grads['b1'] = np.sum(np.dot(dscores ,self.params['W2'].T), axis=0)

        else:
            loss, dz1 = softmax_loss(scores, y)
            loss += 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
            dw1 = np.dot(X.T, dz1) 
            dw1 = np.reshape(dw1, (self.input_dim, self.num_classes))
            dw1 += self.reg * self.params['W1'] 

            db1 = np.sum(dz1, axis=0, keepdims=True)
            grads['W1'] = dw1
            grads['b1'] = db1

            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################
        return loss, grads