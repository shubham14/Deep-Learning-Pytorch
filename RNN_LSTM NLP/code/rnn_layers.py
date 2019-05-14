"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
import numpy as np
import numpy.ma as ma

np.set_printoptions(suppress=True)

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################

    next_h = np.tanh(np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b)
    cache = x, prev_h, Wx, Wh, b, next_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b, next_h = cache
    N, D = x.shape
    dx = np.matmul(np.multiply(dnext_h, (1 - next_h**2)), Wx.T)
    dWx = np.matmul(x.T, np.multiply((1 - next_h ** 2), dnext_h))
    db = np.matmul(np.multiply(dnext_h, (1 - next_h ** 2)).T, np.ones(N))
    dWh = np.matmul(prev_h.T, np.multiply((1 - next_h ** 2), dnext_h))
    dprev_h = np.matmul(np.multiply(dnext_h, (1 - next_h ** 2)), Wh.T)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    T, N, _ = x.shape
    _, H = h0.shape
    h = np.zeros((T, N, H))
    curr_h = h0
    for i in range(T):
        h[i, :, :], _ = rnn_step_forward(x[i, :, :], curr_h, Wx, Wh, b)
        curr_h = h[i, :, :]
    cache = x, Wx, Wh, b, h, h0
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    x, Wx, Wh, b, h, h0 = cache
    T, N, H = dh.shape
    _, _, D = x.shape
    dh0 = np.zeros((N, H))
    dx = np.zeros_like(x)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(dh0)

    for t in range(T, 0, -1):
        if t > 1:
            cache = x[t-1, :, :], h[t-2, :, :], Wx, Wh, b, h[t-1, :, :]
            dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dh[t-1, :, :] + dprev_h, cache)
        else:
            cache = x[t-1, :, :], h0, Wx, Wh, b, h[t-1, :, :]
            dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dh[t-1, :, :] + dprev_h, cache)
            dh0 = dprev_h

        dx[t-1, :, :] = dx_
        dWx += dWx_
        dWh += dWh_
        db += db_

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################
    N, H = prev_h.shape
    A = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b
    f_t, x_i, c_t, o_t = sigmoid(A[:,0:H]), sigmoid(A[:, H:2*H]), np.tanh(A[:, 2*H:3*H]), sigmoid(A[:, 3*H:4*H])
    next_c = np.multiply(f_t, prev_c) + np.multiply(x_i, c_t)
    next_h = np.multiply(o_t, np.tanh(next_c))
    cache = x, prev_h, prev_c, Wx, Wh, b, next_h, next_c
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db, dprev_h, dprev_c = None, None, None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, next_h, next_c = cache
    A = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b
    N, D = x.shape
    _, H = dnext_c.shape

    f_t, i_t, c_t, o_t = sigmoid(A[:,0:H]), sigmoid(A[:, H:2*H]), np.tanh(A[:, 2*H:3*H]), sigmoid(A[:, 3*H:4*H])

    Wxf, Wxi, Wxc, Wxo = Wx[:, 0:H], Wx[:, H:2*H], Wx[:, 2*H:3*H], Wx[:, 3*H:4*H]
    Whf, Whi, Whc, Who = Wh[:, 0:H], Wh[:, H:2*H], Wh[:, 2*H:3*H], Wh[:, 3*H:4*H]

    # c-track
    dct_dbf_c = np.sum(np.multiply(np.multiply(np.multiply(f_t , (1 - f_t)), prev_c), dnext_c), axis=0)
    dct_dbi_c = np.sum(np.multiply(np.multiply(np.multiply(i_t , (1 - i_t)), c_t), dnext_c), axis=0)
    dct_dbc_c = np.sum(np.multiply(np.multiply(1 - c_t ** 2, i_t), dnext_c), axis=0)
    dct_dbo_c = np.zeros_like(dct_dbc_c)

    db_c = np.hstack((dct_dbf_c, dct_dbi_c, dct_dbc_c, dct_dbo_c))

    dct_dWfx_c = np.matmul(x.T, np.multiply(np.multiply(np.multiply(f_t , (1 - f_t)), dnext_c), prev_c))
    dct_dWix_c = np.matmul(x.T, np.multiply(np.multiply(np.multiply(i_t , (1 - i_t)), dnext_c), c_t))
    dct_dWcx_c = np.matmul(x.T, np.multiply(np.multiply(1 - c_t**2, dnext_c), i_t))
    dct_dWox_c = np.zeros_like(dct_dWfx_c)

    dWx_c = np.hstack((dct_dWfx_c, dct_dWix_c, dct_dWcx_c, dct_dWox_c))

    dct_dWfh_c = np.matmul(prev_h.T, np.multiply(np.multiply(np.multiply(f_t , (1 - f_t)), dnext_c), prev_c))
    dct_dWih_c = np.matmul(prev_h.T, np.multiply(np.multiply(np.multiply(i_t , (1 - i_t)), dnext_c), c_t))
    dct_dWch_c = np.matmul(prev_h.T, np.multiply(np.multiply(1 - c_t**2, dnext_c), i_t))
    dct_dWoh_c = np.zeros_like(dct_dWfh_c)

    dWh_c = np.hstack((dct_dWfh_c, dct_dWih_c, dct_dWch_c, dct_dWoh_c))
    
    dprev_c_c = np.multiply(dnext_c, f_t)

    dx1_c = np.matmul(np.multiply(np.multiply(dnext_c, prev_c), np.multiply(f_t, 1 - f_t)), Wxf.T)
    dx2_c = np.matmul(np.multiply(np.multiply(dnext_c, c_t), np.multiply(i_t, 1 - i_t)), Wxi.T)
    dx3_c = np.matmul(np.multiply(np.multiply(dnext_c, i_t), 1 - c_t**2), Wxc.T)

    dx_c = dx1_c + dx2_c + dx3_c

    dh1_c = np.matmul(np.multiply(np.multiply(dnext_c, prev_c), np.multiply(f_t, 1 - f_t)), Whf.T)
    dh2_c = np.matmul(np.multiply(np.multiply(dnext_c, c_t), np.multiply(i_t, 1 - i_t)), Whi.T)
    dh3_c = np.matmul(np.multiply(np.multiply(dnext_c, i_t), 1 - c_t**2), Whc.T)

    dprev_h_c = dh1_c + dh2_c + dh3_c

    # h-track
    o_t_tanh = np.multiply(o_t, (1 - np.tanh(next_c)**2))

    dht_dbf_h = np.sum(np.multiply(np.multiply(np.multiply(o_t_tanh, prev_c), dnext_h), np.multiply(f_t, 1 - f_t)), axis=0)
    dht_dbi_h = np.sum(np.multiply(np.multiply(np.multiply(o_t_tanh, c_t), dnext_h), np.multiply(i_t, 1 - i_t)), axis=0)
    dht_dbc_h = np.sum(np.multiply(np.multiply(np.multiply(o_t_tanh, i_t), dnext_h), 1 - c_t ** 2), axis=0)
    dht_dbo_h = np.sum(np.multiply(np.multiply(np.multiply(o_t, 1 - o_t), np.tanh(next_c)), dnext_h), axis=0)

    db_h = np.hstack((dht_dbf_h, dht_dbi_h, dht_dbc_h, dht_dbo_h))
    dprev_c_h = np.multiply(np.multiply(o_t_tanh, f_t), dnext_h)

    dx1_h = np.matmul(np.multiply(np.multiply(np.multiply(dnext_h, prev_c), np.multiply(f_t, 1 - f_t)), o_t_tanh), Wxf.T)
    dx2_h = np.matmul(np.multiply(np.multiply(np.multiply(dnext_h, c_t), np.multiply(i_t, 1 - i_t)), o_t_tanh), Wxi.T)
    dx3_h = np.matmul(np.multiply(np.multiply(np.multiply(dnext_h, i_t), 1 - c_t**2), o_t_tanh), Wxc.T)
    dx4_h = np.matmul(np.multiply(np.multiply(np.multiply(o_t, 1 - o_t), np.tanh(next_c)), dnext_h), Wxo.T)

    dx_h = dx1_h + dx2_h + dx3_h + dx4_h

    dh1_h = np.matmul(np.multiply(np.multiply(np.multiply(dnext_h, prev_c), np.multiply(f_t, 1 - f_t)), o_t_tanh), Whf.T)
    dh2_h = np.matmul(np.multiply(np.multiply(np.multiply(dnext_h, c_t), np.multiply(i_t, 1 - i_t)), o_t_tanh), Whi.T)
    dh3_h = np.matmul(np.multiply(np.multiply(np.multiply(dnext_h, i_t), 1 - c_t**2), o_t_tanh), Whc.T)
    dh4_h = np.matmul(np.multiply(np.multiply(np.multiply(o_t, 1 - o_t), np.tanh(next_c)), dnext_h), Who.T)

    dprev_h_h =  dh1_h + dh2_h + dh3_h + dh4_h

    dct_dWfx_h = np.matmul(x.T, np.multiply(np.multiply(np.multiply(np.multiply(f_t , (1 - f_t)), dnext_h), prev_c), o_t_tanh))
    dct_dWix_h = np.matmul(x.T, np.multiply(np.multiply(np.multiply(np.multiply(i_t , (1 - i_t)), dnext_h), c_t), o_t_tanh))
    dct_dWcx_h = np.matmul(x.T, np.multiply(np.multiply(np.multiply(1 - c_t ** 2, dnext_h), i_t), o_t_tanh))
    dct_dWox_h = np.matmul(x.T, np.multiply(np.multiply(np.multiply(o_t, 1 - o_t), np.tanh(next_c)), dnext_h))

    dWx_h = np.hstack((dct_dWfx_h, dct_dWix_h, dct_dWcx_h, dct_dWox_h))

    dct_dWfh_h = np.matmul(prev_h.T, np.multiply(np.multiply(np.multiply(np.multiply(f_t , (1 - f_t)), dnext_h), prev_c), o_t_tanh))
    dct_dWih_h = np.matmul(prev_h.T, np.multiply(np.multiply(np.multiply(np.multiply(i_t , (1 - i_t)), dnext_h), c_t), o_t_tanh))
    dct_dWch_h = np.matmul(prev_h.T, np.multiply(np.multiply(np.multiply(1 - c_t ** 2, dnext_h), i_t), o_t_tanh))
    dct_dWoh_h = np.matmul(prev_h.T, np.multiply(np.multiply(np.multiply(o_t, 1 - o_t), np.tanh(next_c)), dnext_h))

    dWh_h = np.hstack((dct_dWfh_h, dct_dWih_h, dct_dWch_h, dct_dWoh_h))

    dx = dx_c + dx_h
    dprev_c = dprev_c_c + dprev_c_h
    dprev_h = dprev_h_c + dprev_h_h
    dWx = dWx_c + dWx_h
    dWh = dWh_c + dWh_h
    db = db_c + db_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    T, N, _ = x.shape
    _, H = h0.shape
    h = np.zeros((T, N, H))
    c0 = np.zeros_like(h)
    prev_h = h0
    c = np.zeros_like(h)
    prev_c = c[0, :, :]
    for i in range(T):
        h[i, :, :], c[i, :, :], cache_temp = lstm_step_forward(x[i, :, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = h[i, :, :]
        prev_c = c[i, :, :]
    cache = x, h, Wx, Wh, h0, b, c
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, tuple(cache)


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    T, N, H = dh.shape
    x, h, Wx, Wh, h0, b, c= cache
    D = x.shape[2]
    dx = np.zeros((T, N, D))
    db = np.zeros((4 * H))
    db_temp = np.zeros((4 * H, T))
    dh0 = np.zeros((N, H))
    c0 = c[0, :, :].copy()
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    d_prev_h = np.zeros_like(dh0)
    d_prev_c = np.zeros_like(c0)
    for t in range(T,0,-1):
        next_h = h[t-1, :, :]
        next_c = c[t-1, :, :]
        next_x = x[t-1, :, :] 
        if t != 1:
            prev_h = h[t-2, :, :]
            prev_c = c[t-2,:,:]
        elif t == 1:
            prev_h = h0
            prev_c = np.zeros_like(c0)

        temp_cache = next_x, prev_h, prev_c, Wx, Wh, b, next_h, next_c
        dx_temp, d_prev_h, d_prev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(dh[t-1,:,:] + d_prev_h, d_prev_c, temp_cache)
       
        dx[t-1, :, :] = dx_temp
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
    dh0 = d_prev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW


def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]

    out = x.dot(w) + b
    cache = x, w, b, out
    return out, cache



def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    D, M = w.shape
    N, T, _ = x.shape
    dx = dout.reshape(N * T, M).dot(w.T)
    dx = dx.reshape(N, T, D)
    dw = x.reshape(N * T, D).T.dot(dout.reshape(N * T, M))
    db = dout.reshape(N * T, M).T.dot(np.ones(N*T))
    return dx, dw, db


def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape
    loss = 0
    dx = np.zeros((N, T, V))
    for t in range(T):    
        x_new = x[:, t, :]
        y_new = y[:, t]
        mask_new = mask[:, t]
        probs = np.exp(x_new)
        probs /= np.sum(probs, axis=1, keepdims=True)
        ll = np.multiply(np.log(probs[np.arange(N), y_new]), mask_new)
        loss += -np.mean(ll)
        dx_new = probs
        dx_new[np.arange(N), y_new] -= 1
        dx_new1 = np.multiply(dx_new, mask_new.astype(int).reshape(-1, 1))
        dx[:, t, :] = dx_new1
    
    return loss, dx/N
