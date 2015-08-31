import numpy as np
import matplotlib.pyplot as plt
from theano import *
import theano.tensor as T
from theano import function

x = T.dmatrix('x')
theta = T.dmatrix('theta')
y = T.dmatrix('y')

z = 1/(1+T.exp(-T.dot(theta.T,x)))
f = function([x,theta], z)

xtest = np.arange(1,11,1)
xtest.shape = (2,5)

thetatest = np.arange(0.1,1.1,.1)
thetatest.shape = (2,len(thetatest)/2)

print f(xtest, thetatest)
print '---'
print  thetatest
print xtest
