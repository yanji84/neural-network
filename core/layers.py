import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  out = x.reshape(x.shape[0], -1).dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  dx = dout.dot(w.T).reshape(x.shape)
  dw = dout.T.dot(x.reshape(x.shape[0], -1)).T
  db = np.sum(dout, axis = 0)
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
  out = np.maximum(0, x)
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
  dx = dout
  dx[x <= 0] = 0
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
    # step 1 - calculate mean (D,)
    mu = 1./N * np.sum(x, axis=0)

    # step 2 - subtract mean from input (N,D)
    xmu = x - mu

    # step 3 - square xmu (N,D)
    sqr = xmu ** 2

    # step 4 - calculate var (D,)
    var = 1./N * np.sum(sqr, axis = 0)

    # step 5 - calculate std (D,)
    std = np.sqrt(var + eps)

    # step 6 - inverse std (D,)
    istd = 1. / std

    # step 7 - calculate normalized input (N,D)
    normalized = xmu * istd

    # step 8 - include gamma
    scaled = normalized * gamma

    # step 9 - include beta
    out = scaled + beta

    cache = (normalized, gamma, istd, xmu, std, var, eps)

    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var
  elif mode == 'test':
    out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
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
  
  normalized, gamma, istd, xmu, std, var, eps = cache

  # step 9
  N,D = dout.shape
  dbeta = np.sum(dout, axis=0)
  dscaled = dout

  # step 8
  dgamma = np.sum(dscaled * normalized, axis=0)
  dnormalized = dscaled * gamma

  # step 7
  dxmu1 = dnormalized * istd
  distd = np.sum(dnormalized * xmu, axis=0)

  # step 6
  dstd = distd * -1./(std ** 2)

  # step 5
  dvar = dstd * 0.5 * 1. / np.sqrt(var + eps)

  # step 4
  dsqr = dvar * np.ones((N,D)) * 1. / N

  # step 3
  dxmu2 = dsqr * 2 * xmu

  # step 2
  dxmu = dxmu1 + dxmu2
  dx1 = dxmu
  dmu = -1 * np.sum(dxmu, axis=0)

  # step 1
  dx2 = dmu * 1./N * np.ones((N,D))
  dx = dx1 + dx2

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = np.random.rand(*x.shape) < dropout_param['p']
    mask = mask.astype(x.dtype)
    mask /= dropout_param['p']
    out = x * mask
  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None

  stride = conv_param['stride']
  pad = conv_param['pad']
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  numimages,imagedepth,imageh,imagew = x.shape
  numfilters,_,filterh,filterw = w.shape
  nexth = 1 + (imageh + 2 * pad - filterh) / stride
  nextw = 1 + (imagew + 2 * pad - filterw) / stride
  out = np.zeros((numimages, numfilters, nexth, nextw))
  for n in range(numimages): 
    for i in range(nextw):
      for j in range(nexth):
        iw = i * stride
        ih = j * stride
        receptive_field = padded[n, :, ih:ih+filterh, iw:iw+filterw]
        out[n,range(numfilters),j,i] = np.sum(receptive_field * w, axis=(1,2,3)) + b
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']

  _,imagedepth,imageh,imagew = x.shape
  numimages,numfilters,outh,outw = dout.shape
  _,_,filterh,filterw = w.shape
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  dpadded = np.zeros(padded.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  for n in range(numimages):
    for f in range(numfilters):
      for i in range(outh):
        for j in range(outw):
          dreceptive_field = dout[n,f,i,j] * w[f]
          pi = i*stride
          pj = j*stride
          dpadded[n,:,pi:pi+filterh,pj:pj+filterw] += dreceptive_field
          dw[f] = dw[f] + dout[n,f,i,j] * padded[n,:,pi:pi+filterh,pj:pj+filterw]
          db[f] = db[f] + dout[n,f,i,j]
  dx = dpadded[:,:,pad:pad+imageh,pad:pad+imagew]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  poolheight = pool_param['pool_height']
  poolwidth = pool_param['pool_width']
  stride = pool_param['stride']
  numimages,imagedepth,imageh,imagew = x.shape
  outh = 1 + (imageh - poolheight) / stride
  outw = 1 + (imagew - poolwidth) / stride
  out = np.zeros((numimages, imagedepth, outh, outw))

  for n in range(numimages):
    for c in range(imagedepth):
      for i in range(outh):
        for j in range(outw):
          winh = i * stride
          winw = j * stride
          window = x[n,c,winh:winh+poolheight,winw:winw+poolwidth]
          out[n,c,i,j] = np.amax(window)

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  poolheight = pool_param['pool_height']
  poolwidth = pool_param['pool_width']
  stride = pool_param['stride']
  numimages,imagedepth,imageh,imagew = x.shape
  _,_,outh,outw = dout.shape
  dx = np.zeros((numimages,imagedepth,imageh,imagew))
  for n in range(numimages):
    for c in range(imagedepth):
      for i in range(outh):
        for j in range(outw):
          winh = i * stride
          winw = j * stride
          wineh = winh+poolheight
          winew = winw+poolwidth
          window = x[n,c,winh:winh+poolheight,winw:winw+poolwidth]
          m = np.max(window)
          dx[n,c,winh:wineh,winw:winew] += (window==m)*dout[n,c,i,j]
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  pass
  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None
  pass
  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
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
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
