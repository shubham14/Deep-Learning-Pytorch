import numpy as np

from rnn_layers import *


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




#rnn_step_forward
np.random.seed(498)
N, D, H = 5, 10, 8
x = np.linspace(-0.1, 0.5, num=N * D).reshape(N, D)
prev_h = np.linspace(-0.1, 0.5, num=N * H).reshape(N, H)
Wx = np.linspace(-0.2, 0.3, num=D * H).reshape(D, H)
Wh = np.linspace(-0.2, 0.3, num=H * H).reshape(H, H)
b = np.linspace(-0.3, 0.1, num=H)

out, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
correct_out = np.array([[-0.22463597, -0.17533181, -0.12513221, -0.07428362, -0.02304582,  0.02831334,
   0.07952345,  0.13031722],
 [-0.17086067, -0.10522695, -0.03866349,  0.02824486,  0.09490108,  0.16071707,
   0.22513414,  0.28764139],
 [-0.11604783, -0.03406035,  0.04838815,  0.13018247,  0.21024242,  0.28757852,
   0.36133798,  0.43083656],
 [-0.06051892,  0.03745334,  0.13471086,  0.22944027,  0.32001372,  0.40509249,
   0.48369,     0.55519071],
 [-0.00461288,  0.10858546,  0.21903615,  0.32414482,  0.42181666,  0.51059038,
   0.58966751,  0.65885315]])

# Compare your output with ours. The error should be around 1e-7.
print('Testing rnn_step_forward function:')
print('difference: ', rel_error(out, correct_out))



#rnn_step_backward
np.random.seed(498)
N, D, H = 6, 7, 8
x = np.random.randn(N, D)
prev_h = np.random.randn(N, H)
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(H,)

dnext_h = np.random.randn(N, H)

dx_num = eval_numerical_gradient_array(lambda x: rnn_step_forward(x, prev_h, Wx, Wh, b)[0], x, dnext_h)
dprev_h_num = eval_numerical_gradient_array(lambda prev_h: rnn_step_forward(x, prev_h, Wx, Wh, b)[0], prev_h, dnext_h)
dWx_num = eval_numerical_gradient_array(lambda Wx: rnn_step_forward(x, prev_h, Wx, Wh, b)[0], Wx, dnext_h)
dWh_num = eval_numerical_gradient_array(lambda Wh: rnn_step_forward(x, prev_h, Wx, Wh, b)[0], Wh, dnext_h)
db_num = eval_numerical_gradient_array(lambda b: rnn_step_forward(x, prev_h, Wx, Wh, b)[0], b, dnext_h)

_, cache = rnn_step_forward(x, prev_h, Wx, Wh, b)
dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

# The errors should be around 2e-9 to 1e-10
print('\nTesting rnn_step_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))





#rnn_forward
T, N, D, H = 2, 3, 4, 5
np.random.seed(498)
x = np.linspace(-0.1, 0.5, num=T * N * D).reshape(T, N, D)
h0 = np.linspace(-0.1, 0.5, num=N * H).reshape(N, H)
Wx = np.linspace(-0.2, 0.3, num=D * H).reshape(D, H)
Wh = np.linspace(-0.2, 0.3, num=H * H).reshape(H, H)
b = np.linspace(-0.3, 0.1, num=H)

out, _ = rnn_forward(x, h0, Wx, Wh, b)
correct_out = np.array([[[-0.23374681, -0.14501505, -0.05388776,  0.03814605,  0.12953723],
  [-0.22633104, -0.10452583,  0.02049811,  0.14488423,  0.26485357],
  [-0.21888893, -0.06368711,  0.09465773,  0.24835356,  0.39048552]],

 [[-0.19057771, -0.07178839,  0.04907294,  0.16851404,  0.28320339],
  [-0.15616353, -0.01785099,  0.12114815,  0.25555323,  0.38075649],
  [-0.12383497,  0.03342045,  0.18903833,  0.33570708,  0.46772415]]])

# Compare your output with ours. The error should be around 6e-8.
print('\nTesting rnn_forward function:')
print('difference: ', rel_error(out, correct_out))



#rnn_backward
np.random.seed(498)
T, N, D, H = 4, 5, 6, 7
x = np.random.randn(T, N, D)
h0 = np.random.randn(N, H)
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(H,)

dh = np.random.randn(T, N, H)


dx_num = eval_numerical_gradient_array(lambda x: rnn_forward(x, h0, Wx, Wh, b)[0], x, dh)
dh0_num = eval_numerical_gradient_array(lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0], h0, dh)
dWx_num = eval_numerical_gradient_array(lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0], Wx, dh)
dWh_num = eval_numerical_gradient_array(lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0], Wh, dh)
db_num = eval_numerical_gradient_array(lambda b: rnn_forward(x, h0, Wx, Wh, b)[0], b, dh)

_, cache = rnn_forward(x, h0, Wx, Wh, b)
dx, dh0, dWx, dWh, db = rnn_backward(dh, cache)

# The errors should be around 2e-9 to 1e-10
print('\nTesting rnn_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dh0 error: ', rel_error(dh0_num, dh0))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))



#lstm_step_forward
np.random.seed(498)
N, D, H = 3, 5, 4
x = np.linspace(-0.1, 0.5, num=N * D).reshape(N, D)
prev_h = np.linspace(-0.1, 0.5, num=N * H).reshape(N, H)
prev_c = np.linspace(-0.8, 0.8, num=N * H).reshape(N, H)
Wx = np.linspace(-0.5, 0.3, num=4 * D * H).reshape(D, 4 * H)
Wh = np.linspace(-0.6, 0.2, num=4 * H * H).reshape(H, 4 * H)
b = np.linspace(-0.8, 0.6, num=4 * H)

next_h, next_c, _ = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
correct_next_h = np.array([[-0.14212115, -0.09868166, -0.04508843,  0.01826009],
 [-0.06578861, -0.02624557,  0.02564223,  0.0897345 ],
 [-0.02585077,  0.00574838,  0.05104366,  0.11121977]])

correct_next_c = np.array([[-0.23621572, -0.15684323, -0.0689127,   0.02702961],
 [-0.11622365, -0.04404816,  0.04116592,  0.13904129],
 [-0.04944276,  0.010322,    0.08672159,  0.18063112]])

# Compare your output with ours. The error should be around 1e-7.
print('\nTesting lstm_step_forward function:')
print('next_h error: ', rel_error(next_h, correct_next_h))
print('next_c error: ', rel_error(next_c, correct_next_c))







#lstm_step_backward (dnext_h track)
np.random.seed(498)
N, D, H = 6, 7, 8
x = np.random.randn(N, D)
prev_h = np.random.randn(N, H)
prev_c = np.random.randn(N, H)
Wx = np.random.randn(D, 4 * H)
Wh = np.random.randn(H, 4 * H)
b = np.random.randn(4 * H,)

dnext_h = np.random.randn(N, H)
dnext_c = np.zeros((N, H))

dx_num = eval_numerical_gradient_array(lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0], x, dnext_h)
dprev_h_num = eval_numerical_gradient_array(lambda prev_h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0], prev_h, dnext_h)
dprev_c_num = eval_numerical_gradient_array(lambda prev_c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0], prev_c, dnext_h)
dWx_num = eval_numerical_gradient_array(lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0], Wx, dnext_h)
dWh_num = eval_numerical_gradient_array(lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0], Wh, dnext_h)
db_num = eval_numerical_gradient_array(lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0], b, dnext_h)

_, _, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache)

# The errors should be around 3e-8 to 4e-10
print('\nTesting lstm_step_backward (dnext_h track) function:')
print('dx error: ', rel_error(dx_num, dx))
print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
print('dprev_c error: ', rel_error(dprev_c_num, dprev_c))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))


#lstm_step_backward (dnext_c track)
np.random.seed(498)
N, D, H = 6, 7, 8
x = np.random.randn(N, D)
prev_h = np.random.randn(N, H)
prev_c = np.random.randn(N, H)
Wx = np.random.randn(D, 4 * H)
Wh = np.random.randn(H, 4 * H)
b = np.random.randn(4 * H,)

dnext_h = np.zeros((N, H))
dnext_c = np.random.randn(N, H)

dx_num = eval_numerical_gradient_array(lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1], x, dnext_c)
dprev_h_num = eval_numerical_gradient_array(lambda prev_h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1], prev_h, dnext_c)
dprev_c_num = eval_numerical_gradient_array(lambda prev_c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1], prev_c, dnext_c)
dWx_num = eval_numerical_gradient_array(lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1], Wx, dnext_c)
dWh_num = eval_numerical_gradient_array(lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1], Wh, dnext_c)
db_num = eval_numerical_gradient_array(lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1], b, dnext_c)

_, _, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache)

# The errors should be around 5e-8 to 3e-10
print('\nTesting lstm_step_backward (dnext_c track) function:')
print('dx error: ', rel_error(dx_num, dx))
print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
print('dprev_c error: ', rel_error(dprev_c_num, dprev_c))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))








#lstm_forward
T, N, D, H = 2, 3, 4, 5
np.random.seed(498)
x = np.linspace(-0.1, 0.5, num=T * N * D).reshape(T, N, D)
h0 = np.linspace(-0.1, 0.5, num=N * H).reshape(N, H)
Wx = np.linspace(-0.2, 0.3, num=4 * D * H).reshape(D, 4 * H)
Wh = np.linspace(-0.2, 0.3, num=4 * H * H).reshape(H, 4 * H)
b = np.linspace(-0.3, 0.1, num=4 * H)

out, cache = lstm_forward(x, h0, Wx, Wh, b)

correct_out = np.array([[[-0.01102618, -0.00660349, -0.00199597,  0.00279512,  0.00776809],
  [ 0.00824255,  0.01562256,  0.02335621,  0.03143392,  0.03984436],
  [ 0.03038912,  0.04127036,  0.05268392,  0.06459734,  0.07697417]],

 [[-0.00858108,  0.00012034,  0.00935358,  0.01911405,  0.02939425],
  [ 0.00784412,  0.01924491,  0.03136921,  0.04420177,  0.05772152],
  [ 0.02587248,  0.04028845,  0.05562386,  0.07184049,  0.08888991]]])

# Compare your output with ours. The error should be around 2e-5.
print('\nTesting lstm_forward function:')
print('difference: ', rel_error(out, correct_out))




#lstm_backward
np.random.seed(498)
T, N, D, H = 4, 6, 7, 5
x = np.random.randn(T, N, D)
h0 = np.random.randn(N, H)
Wx = np.random.randn(D, 4 * H)
Wh = np.random.randn(H, 4 * H)
b = np.random.randn(4 * H,)

dh = np.random.randn(T, N, H)


dx_num = eval_numerical_gradient_array(lambda x: lstm_forward(x, h0, Wx, Wh, b)[0], x, dh)
dh0_num = eval_numerical_gradient_array(lambda h0: lstm_forward(x, h0, Wx, Wh, b)[0], h0, dh)
dWx_num = eval_numerical_gradient_array(lambda Wx: lstm_forward(x, h0, Wx, Wh, b)[0], Wx, dh)
dWh_num = eval_numerical_gradient_array(lambda Wh: lstm_forward(x, h0, Wx, Wh, b)[0], Wh, dh)
db_num = eval_numerical_gradient_array(lambda b: lstm_forward(x, h0, Wx, Wh, b)[0], b, dh)

_, cache = lstm_forward(x, h0, Wx, Wh, b)
dx, dh0, dWx, dWh, db = lstm_backward(dh, cache)

print(dh0)
print('---------------')
print(dh0_num)

# The errors should be around 1e-8 to 5e-10
print('\nTesting lstm_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dh0 error: ', rel_error(dh0_num, dh0))
print('dWx error: ', rel_error(dWx_num, dWx))
print('dWh error: ', rel_error(dWh_num, dWh))
print('db error: ', rel_error(db_num, db))


#temporal_fc_forward
np.random.seed(498)
N, T, D, M = 3, 2, 5, 4
x = np.linspace(-0.1, 0.5, num=N * T * D).reshape(N, T, D)
w = np.linspace(-0.1, 0.5, num=D * M).reshape(D, M)
b = np.linspace(-0.3, 0.1, num=M)

out, _ = temporal_fc_forward(x, w, b)
correct_out = np.array([[[-0.31860254, -0.19452511, -0.07044767,  0.05362976],
  [-0.23965517, -0.0992438,   0.04116757,  0.18157895]],

 [[-0.1607078,  -0.00396249,  0.15278282,  0.30952813],
  [-0.08176044,  0.09131881,  0.26439806,  0.43747731]],

 [[-0.00281307,  0.18660012,  0.37601331,  0.5654265 ],
  [ 0.0761343,   0.28188143,  0.48762855,  0.69337568]]])

# Compare your output with ours. The error should be around 5e-7.
print('\nTesting temporal_fc_forward function:')
print('difference: ', rel_error(out, correct_out))



#temporal_fc_backward
np.random.seed(498)
N, T, D, M = 3, 2, 5, 4
x = np.random.randn(N, T, D)
w = np.random.randn(D, M)
b = np.random.randn(M,)
dout = np.random.randn(N, T, M)

dx_num = eval_numerical_gradient_array(lambda x: temporal_fc_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: temporal_fc_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: temporal_fc_forward(x, w, b)[0], b, dout)

_, cache = temporal_fc_forward(x, w, b)
dx, dw, db = temporal_fc_backward(dout, cache)



# The error should be around 1e-10
print('\nTesting temporal_fc_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


#temporal_softmax_loss
np.random.seed(498)
N, T, V = 10, 6, 8
x = 0.001 * np.random.randn(N, T, V)
y = np.random.randint(V, size=(N, T))
mask = np.random.randint(2, size=(N, T))

dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)
loss, dx = temporal_softmax_loss(x, y, mask)



# Test softmax_loss function. Loss should be 5.4 and dx error should be 2e-9
print('\nTesting temporal_softmax_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))
