import numpy as np

from layers import *


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))



def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval



        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


#FC layer: foward
num_inputs = 2
dim = 120
output_dim = 3

input_size = num_inputs * dim
weight_size = output_dim * dim

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, dim)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(dim, output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

out, _ = fc_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

# Compare your output with ours. The error should be around 1e-9.
print('Testing fc_forward function:')
print('difference: ', rel_error(out, correct_out))



#FC layer: backward
np.random.seed(498)
x = np.random.randn(10, 6)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: fc_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: fc_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: fc_forward(x, w, b)[0], b, dout)

_, cache = fc_forward(x, w, b)
dx, dw, db = fc_backward(dout, cache)

# The error should be around 1e-10
print('\nTesting fc_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))



# ReLU layer: forward
x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

out, _ = relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

# Compare your output with ours. The error should be around 5e-8
print('\nTesting relu_forward function:')
print('difference: ', rel_error(out, correct_out))


# ReLU layer: backward
np.random.seed(498)
x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)

dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

_, cache = relu_forward(x)
dx = relu_backward(dout, cache)
# The error should be around 3e-12
print('\nTesting relu_backward function:')
print('dx error: ', rel_error(dx_num, dx))





# Softmax loss
np.random.seed(498)
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)



dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)

# Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
print('\nTesting softmax_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))




# Logistic loss (for {-1,1} labels), it's enough to pass one test from Logistic loss and Logistic loss alternative
np.random.seed(498)
num_classes, num_inputs = 1, 50
x = 0.001 * np.random.randn(num_inputs,)
y = np.random.randint(num_classes + 1, size=num_inputs)
for ite in range(num_inputs):
    if y[ite] == 0:
        y[ite] = -1

dx_num = eval_numerical_gradient(lambda x: logistic_loss(x, y)[0], x, verbose=False)
loss, dx = logistic_loss(x, y)

# Test logistic_loss function. Loss should be 0.693 and dx error should be 6e-10
print('\nTesting logistic_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))


# Logistic loss alternative (for {0,1} labels)
np.random.seed(498)
num_classes, num_inputs = 1, 50
x = 0.001 * np.random.randn(num_inputs,)
y = np.random.randint(num_classes + 1, size=num_inputs)


dx_num = eval_numerical_gradient(lambda x: logistic_loss(x, y)[0], x, verbose=False)
loss, dx = logistic_loss(x, y)

# Test logistic_loss function. Loss should be 0.693 and dx error should be 6e-10
print('\nTesting logistic_loss_alternative:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))



# SVM loss (for {-1,1} labels), it's enough to pass one test from SVM loss and SVM loss alternative
np.random.seed(498)
num_classes, num_inputs = 1, 50
x = 0.001 * np.random.randn(num_inputs,)
y = np.random.randint(num_classes + 1, size=num_inputs)
for ite in range(num_inputs):
    if y[ite] == 0:
        y[ite] = -1

dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
loss, dx = svm_loss(x, y)

# Test svm_loss function. Loss should be 1.000 and dx error should be 3e-10
print('\nTesting svm_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))


# SVM loss alternative (for {0,1} labels)
np.random.seed(498)
num_classes, num_inputs = 1, 50
x = 0.001 * np.random.randn(num_inputs,)
y = np.random.randint(num_classes + 1, size=num_inputs)


dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
loss, dx = svm_loss(x, y)

# Test svm_loss function. Loss should be 1.000 and dx error should be 3e-10
print('\nTesting svm_loss_alternative:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))



# Batchnorm foward
x = np.linspace(-0.1, 0.5, num=6).reshape(2, 3)
gamma = np.linspace(-0.2, 0.3, num=3)
beta = np.linspace(-0.3, 0.1, num=3)
bn_param = {
    'mode': 'train',
    'eps': 1e-5,
    'momentum': 0.9
}

out, _ = batchnorm_forward(x, gamma, beta, bn_param)
correct_out = np.array([[-0.10003086, -0.14999229, -0.19995371],
                        [-0.49996914, -0.05000771, 0.39995371]])

# Compare your output with ours. The error should be around 4e-8.
print('\nTesting batchnorm_forward function:')
print('difference: ', rel_error(out, correct_out))




# batchnorm backward
np.random.seed(498)
x = np.random.randn(10, 5)
gamma = np.random.randn(5)
beta = np.random.randn(5)
bn_param = {
    'mode': 'train',
    'eps': 1e-5,
    'momentum': 0.9
}
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0], x, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma: batchnorm_forward(x, gamma, beta, bn_param)[0], gamma, dout)
dbeta_num = eval_numerical_gradient_array(lambda beta: batchnorm_forward(x, gamma, beta, bn_param)[0], beta, dout)

_, cache = batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = batchnorm_backward(dout, cache)

# The errors should be around 1e-10
print('\nTesting batchnorm_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(dgamma_num, dgamma))
print('dbeta error: ', rel_error(dbeta_num, dbeta))




# Dropout foward
np.random.seed(498)
x = np.linspace(-0.1, 0.5, num=6).reshape(2, 3)
dropout_param = {
    'mode': 'train',
    'p': 0.5,
}

out, _ = dropout_forward(x, dropout_param)
correct_out = np.array([[-0.1, 0.02, 0.  ], [ 0., 0., 0. ]])

# Compare your output with ours. The error should be around 2e-16.
print('\nTesting dropout_forward function:')
print('difference: ', rel_error(out, correct_out))



# dropout backward
np.random.seed(498)
x = np.random.randn(10, 5)
dropout_param = {
    'mode': 'train',
    'p': 0.5,
    'seed': 498
}
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: dropout_forward(x, dropout_param)[0], x, dout)

_, cache = dropout_forward(x, dropout_param)
dx = dropout_backward(dout, cache)

# The errors should be around 3e-12
print('\nTesting dropout_backward function:')
print('dx error: ', rel_error(dx_num, dx))



# Conv forward
x = np.linspace(-0.1, 2.5, num=36).reshape(1,1,6,6)
w = np.linspace(-0.9, 0.6, num=9).reshape(1,1,3,3)

out, _ = conv_forward(x, w)
correct_out = np.array([[[[ 1.02085714,  0.92057143,  0.82028571,  0.72      ],
   [ 0.41914286,  0.31885714,  0.21857143,  0.11828571],
   [-0.18257143, -0.28285714, -0.38314286, -0.48342857],
   [-0.78428571, -0.88457143, -0.98485714, -1.08514286]]]])

# Compare your output with ours. The error should be around 2e-8.
print('\nTesting conv_forward function:')
print('difference: ', rel_error(out, correct_out))






# Conv backward
np.random.seed(498)
x = np.random.randn(3, 2, 7, 7)
w = np.random.randn(4, 2, 3, 3)

dout = np.random.randn(3, 4, 5, 5)

dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w)[0], w, dout)

_, cache = conv_forward(x, w)
dx, dw = conv_backward(dout, cache)

print('\nTesting conv_backward function:')
# The errors should be around 3e-9
print('dx error: ', rel_error(dx_num, dx))
# The errors should be around 5e-10
print('dw error: ', rel_error(dw_num, dw))




# max_pool forward
x = np.linspace(-0.1, 2.5, num=49).reshape(1,1,7,7)
pool_param = {
    'pool_height': 3,
    'pool_width': 3,
    'stride': 2
}


out, _ = max_pool_forward(x, pool_param)
correct_out = np.array([[[[0.76666667, 0.875,      0.98333333],
   [1.525,      1.63333333, 1.74166667],
   [2.28333333, 2.39166667, 2.5       ]]]])

# Compare your output with ours. The error should be around 2e-9.
print('\nTesting max_pooling_forward function:')
print('difference: ', rel_error(out, correct_out))







# max_pool backward
np.random.seed(498)
x = np.random.randn(3, 2, 7, 7)
pool_param = {
    'pool_height': 3,
    'pool_width': 3,
    'stride': 2
}

dout = np.random.randn(3, 2, 3, 3)

dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward(x, pool_param)[0], x, dout)


_, cache = max_pool_forward(x, pool_param)
dx = max_pool_backward(dout, cache)

print('\nTesting max_pooling_backward function:')
# The errors should be around 3e-12
print('dx error: ', rel_error(dx_num, dx))