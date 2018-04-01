import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    
    exp_value = np.exp(scores)
    exp_value_sum = np.sum(exp_value)
    norm = exp_value  / exp_value_sum
    if norm[y[i]] == 0:
        continue
    correct_scores_norm = norm[y[i]]
    dW[:,y[i]] += -X[i]
    for j in xrange(num_class):
      dW[:,j] += X[i]*exp_value[j]/exp_value_sum
      
    loss += -1* np.log(correct_scores_norm)
    
  loss/=num_train
  dW/=num_train
  dW+=reg*W*2
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_class = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  margin = np.zeros((num_train,num_class))
  exp_values = np.exp(scores)
  exp_sum = np.sum(exp_values,axis = 1).reshape((num_train,1))
  exp_sum = np.tile(exp_sum, num_class).reshape((num_train,num_class))
  norm = exp_values/ exp_sum
  loss = -np.log(norm)
  loss = np.sum(loss[np.arange(num_train),y])
  loss/= num_train
  margin = norm
  margin[np.arange(num_train),y] -= 1

  dW = np.dot(X.T,margin) 
  dW/=num_train
  dW += 2* reg* W 
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

