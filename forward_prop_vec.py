import numpy as np
import matplotlib.pyplot as plt
from theano import *
import theano.tensor as T
from theano import function

X = T.vector('X')
theta = T.vector('theta')
y = T.dmatrix('y')

z = T.dot(theta.T,X)
#test z = transpose(theta) * X
g = function([X, theta], z)

XX = np.arange(1,11,1)
thetaxx = np.arange(0.1,.6,.05)

zz = g(XX,thetaxx)
print 'theta.T * X = ',zz

##compute activation given input vector=XX and weights=thetaxx
gee = 1/(1+T.exp(-z))
f = function([X, theta], gee)
a = f(XX,thetaxx)
print 'Activationof layer two = g(z2) = a2 =', a
