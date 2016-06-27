import numpy as np

from core.layers import *
from core.fast_layers import *
from core.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
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
    
    numchannels,inputh,inputw = input_dim
    pad = (filter_size - 1) / 2
    stride = 1
    convh = 1 + (inputh - filter_size + 2*pad) / stride
    convw = 1 + (inputw - filter_size + 2*pad) / stride
    poolsize = 2
    poolstride = 2
    poolh = 1 + (convh - poolsize) / poolstride
    poolw = 1 + (convw - poolsize) / poolstride
    w1 = np.random.normal(0,weight_scale,(num_filters,numchannels,filter_size,filter_size))
    b1 = np.zeros(num_filters)
    w2 = np.random.normal(0,weight_scale,(num_filters*poolh*poolw,hidden_dim))
    b2 = np.zeros(hidden_dim)
    w3 = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    b3 = np.zeros(num_classes)

    self.params['W1'] = w1
    self.params['b1'] = b1
    self.params['W2'] = w2
    self.params['b2'] = b2
    self.params['W3'] = w3
    self.params['b3'] = b3

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # layers: conv - relu - 2x2 max pool - affine - relu - affine - softmax
    out,convcache = conv_forward_fast(X, W1, b1, conv_param)
    out,relu1cache = relu_forward(out)
    out,poolcache = max_pool_forward_fast(out, pool_param)
    out,affine1cache = affine_forward(out, W2, b2)
    out,relu2cache = relu_forward(out)
    scores,affine2cache = affine_forward(out, W3, b3)
    
    if y is None:
      return scores

    grads = {}    
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) +
                              np.sum(self.params['W2'] ** 2) +
                              np.sum(self.params['W3'] ** 2))
    dout,grads['W3'],grads['b3'] = affine_backward(dout, affine2cache)
    dout = relu_backward(dout, relu2cache)
    dout,grads['W2'],grads['b2'] = affine_backward(dout, affine1cache)
    dout = max_pool_backward_fast(dout, poolcache)
    dout = relu_backward(dout, relu1cache)
    dout,grads['W1'],grads['b1'] = conv_backward_fast(dout, convcache)
    grads['W1'] += self.reg * grads['W1']
    grads['W2'] += self.reg * grads['W2']
    grads['W3'] += self.reg * grads['W3']

    return loss, grads
  
  
pass
