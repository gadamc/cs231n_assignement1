import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]  #y[i] is a number between 0 and num_classes, indicating the class of the training value
    heaviside = (scores - correct_class_score + 1)>0
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] += -np.sum(np.delete(heaviside,j))*X[i,:]
        continue
      dW[:,j] += heaviside[j]*X[i,:]
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5* reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  S = np.dot(X, W)
  
  #stolen from https://github.com/ethan-mgallagher/cs231/blob/master/cs231n/classifiers/linear_svm.py
  # first example from here: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing
  CS = S[np.arange(num_train), y].reshape(num_train, 1)
    
  L = S - CS + 1  #shape = num_train, num_class
  pos = L > 0
  L = np.multiply(L, pos) #will set zero to all values < 0
  #shape = num_train, num_class
  L = np.sum(L, axis=1) - 1  #sum across all of the class scores (along the num_class axis), then subtract one
  #shape = num_train, 1
  loss = L.sum()/float(num_train) + 0.5 * reg * np.sum(W * W)
  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
 
  boolean = pos * 1.0
  #now count number of j=/=y[i] exceeding margin per training example (row_
  boolean[np.arange(num_train), y] = -(np.sum( boolean, axis=1)-1)
  
  dW = (np.dot(boolean.T,X)).T
  
  #divide by number of training examples
  dW /= float(num_train)
  
  dW += reg*W
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_naive_soln(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    indicator = (scores - correct_class_score + 1)>0
    #print indicator
    for j in xrange(num_classes):  
      if j == y[i]:
        ##do things through matrices, not iteration!!!
        dW[:,j] += -np.sum(np.delete(indicator,j))*X[i,:]
        continue
      #print dW[:,j].size
      #print X[:,i].T.size
      #print indicator[j].size
      dW[:,j] += indicator[j]*X[i,:]
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
       

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #we wanted the average gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized_soln(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
 
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  #correct_scores = X.dot(W)[np.arange(num_train), y] + 1 #gives 1 at correct spot
  correct_scores = X.dot(W)[np.arange(num_train), y]
  #print y[0]
  #print correct_scores.shape
  loss = scores.T - correct_scores.T + 1
  boolean = loss > 0 #every instance where margin is exceeded
  loss = np.sum( loss * boolean, axis=0) - 1
  regularization = 0.5 * reg * np.sum(W * W)
  loss = np.sum(loss) / float(num_train) + regularization
  #print y.shape
  # 500,
  #print X.shape
  #(500, 3072)
  #print scores.shape
  #(500, 10) - ten class scores for each member of training set
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #print boolean.shape 
  # (10, 500)
  
  
  #print dW.shape
  #(3072, 10)
  
  #we are going to build a matrix that has a number of integers
  #in a (10, 500) shape, with the sum of each column representing the number of times X[i]
  #should be applied to get the gradient for that training example
  
  
  #we get a one for every true in boolean matrix
  #e.g. every time margin was exceeded
  boolean = boolean * np.ones( loss.shape )
  #print boolean.shape
  
  #now count number of j=/=y[i] exceeding margin per training example (row_
  boolean[y, np.arange(num_train)] = -(np.sum( boolean, axis=0)-1)
  
  dW = (boolean.dot(X)).T
  
  #divide by number of training examples
  dW /= float(num_train)
  
  temp = reg*W
  #print temp.shape
  dW += temp
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW