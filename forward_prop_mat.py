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

XX = np.arange(0,5,1)
tt = np.arange(0,.25,.01)
tt.shape = (5,5)

a = f(XX, tt)
print 'computed z:\n', tt.T, '*',  XX
print 'z: \n', np.dot(tt.T, XX)
print 'Activations a:\n', a


import matplotlib.pyplot as plt
xplt = np.arange(-len(a),len(a)+1)
plt.plot(xplt, logistic.eval({z:xplt}))
plt.show()
