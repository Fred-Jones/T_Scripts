import numpy as np
import matplotlib.pyplot as plt
from theano import *
import theano.tensor as T
from theano import function

X = T.vector('X')
theta = T.dmatrix('theta')

z = dot(theta.T, X)
logistic = 1/(1+T.exp(-z))

f = function([X, theta], logistic)

XX = np.arange(1,6,1)
tt = np.arange(0,2.5,.1)
tt.shape = (5,5)

a = f(XX, tt)
print 'computing z:\n', tt.T, '*',  XX
print 'Activations a:\n', a

# print tt
# print tt.T
# print np.dot(tt.T,XX)
# print np.dot(XX, tt.T)
