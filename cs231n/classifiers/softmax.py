import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sumexps = np.sum(np.exp(scores))
    loss -= np.log(np.exp(correct_class_score) / sumexps)
    
    for j in range(num_classes):
        dW[:,j] += np.exp(scores[j])*X[i,:]/sumexps
        if j == y[i]:
            dW[:,j] -= X[i,:]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW  /= num_train
  dW += reg*W
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  y[0] = 0
  y[1] = 1
  y[2] = 2
  #y[-3] = 0
  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W)
  print("S")
  print(S.shape)
  # print(S)
  expS = np.exp(S)
  print("expS")
  print(expS.shape)
  # print(expS)
  SumExpS = np.sum(expS, axis=1).reshape(expS.shape[0],1)
  print("SumExpS")
  print(SumExpS.shape)
  # print(SumExpS)
  exp_Syi = expS[np.arange(expS.shape[0]),y].reshape(expS.shape[0],1)

  print("exp_Syi")
  print(exp_Syi.shape)  #should be (num_train,)
  # print(exp_Syi[:5])
  print("y")
  print(y.shape)
  # print(y)

  r_loss = np.divide(exp_Syi,SumExpS)
  r_loss = -1.0*np.log(r_loss)
  print("r_loss")
  print(r_loss.shape)
  # print(r_loss)

  #this is wrong somehow
  #r_loss = r[np.arange(num_train),y]
  #print(r_loss.shape)
  #print(r_loss)
  #r_loss = -1.0*np.log(r_loss).sum(axis=0)
  #print(r_loss.shape)
  #print(r_loss)
   
  loss = r_loss.sum()/num_train + 0.5 * reg * np.sum(W * W)


  r = expS / SumExpS
  r = -1.0*np.log(r)
  print("r")
  print(r.shape)
  print(r)

  delta_yiC = np.zeros_like(r)
  delta_yiC[np.arange(delta_yiC.shape[0]),y] = 1
  print("delta_yiC")
  print(delta_yiC.shape)
  print(delta_yiC)

  r = r - delta_yiC
  print("r - delta_yiC")
  print(r.shape)
  print(r)

  dW = np.dot(X.T, r) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_naive_soln(W, X, y, reg):
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
  
  This SOLUTION IS FROM
  https://github.com/ethan-mgallagher/cs231/blob/master/cs231n/classifiers/softmax.py
  
  It may not be correct!
  
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
  num_classes = W.shape[1]
  
  tester = X.dot(W)
  scores = np.ones( tester.shape )
  #print num_train
  #print num_classes
  
  for r in xrange(num_train):
      #trick
      scores = X[r].dot(W)
      scores -= np.max( scores )
      #get array without score of correct class
      t_arr = scores[ np.arange( scores.size) != y[r]]
      #compute loss
      temp = np.exp( scores[y[r]]) / np.sum( np.exp( t_arr ))
      #if r == 0:
          #print(temp)
      loss -= np.sum( np.log(temp))
      
      #now do derivative
      exp_r = np.exp(scores)
      exp_r_total = np.sum(exp_r)
      for c in xrange(num_classes):
          if c == y[r]:
              dW[:,c] += -X[r].T + (exp_r[c] / exp_r_total ) * X[r].T
          else:
              dW[:,c] += (exp_r[c] / exp_r_total ) * X[r].T
  
  #for r in scores[0].shape()
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= float(num_train)
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= float(num_train)
  dW += reg * W
  
  return loss, dW


def softmax_loss_vectorized_soln(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  
  This SOLUTION IS FROM
  https://github.com/ethan-mgallagher/cs231/blob/master/cs231n/classifiers/softmax.py
  
  It may not be correct!
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  scores = X.dot(W)
  #do trick to prevent instability
  
  trick_arr = np.max( scores, axis=1)
  
  #print trick_arr.shape
  
  scores = scores.T - trick_arr.T
  scores = scores.T
  
  y_arr = np.exp( scores[ np.arange(num_train), y] )
  
  
  
  mask = scores[ np.arange(num_train), y]

  except_y = np.exp(np.copy(scores))
  
  except_y = np.sum( except_y, axis=1) - y_arr 
  
  #print y_arr[0] / except_y[0]
  
  #temp = y_arr / except_y
  #temp = np.log(temp)
  #temp = np.sum( temp, axis = 0)
  
  #loss -= temp
  #loss /= float(num_train)
  #loss += 0.5 * reg * np.sum(W * W)
  
  #now do derivative
  exp_r = np.exp(scores)
  #scores column wise summed
  exp_r_total = np.sum(exp_r, axis=1)
  
  
  
  loss = np.log( exp_r_total ) 
  loss -= np.log(y_arr)
  loss = np.sum(loss) / float(num_train) 
  loss += 0.5 * reg * np.sum(W * W)
  
  grad = exp_r.T / exp_r_total.T
  grad[ y, np.arange(num_train)] -= 1.0
  #grad = grad.T
  dW = grad.dot(X) / float(num_train) + (reg*W).T
  dW = dW.T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
