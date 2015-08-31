from __future__ import division
from theano import *
import theano.tensor as T
from theano import function

#compute y=mx+b == f(x,m,b) and logistic
x = T.dscalar('x')
m = T.dscalar('m')
b = T.scalar('b')

z = m*x+b
l = 1/(1+T.exp(-z))
ll = (1 + T.tanh(z / 2)) / 2

f = function([x,m,b], z)
print 'f', f(2,.5,0)
g = function([x,m,b], l)
print 'g', g(2,.5,0)
g2 = function([x,m,b], ll)
print 'g2', g2(2,.5,0)
##theano pretty aight
l.eval({z:1})
ll.eval({z:1})
l.eval({z:0})
ll.eval({z:0})

##flip around Y axis
import numpy as np
# for i in np.arange(1,11,1):
#     print l.eval({z:-i})
#
# print '--  z=6  --'
# print l.eval({z:6})
def outV(tx):
    out = []
    for i in tx:
        out.append(ll.eval({z:i}))
    return np.array(out)

import matplotlib.pyplot as plt
txx = np.arange(0,7,.5)
neg_txx = -1 * txx
nxx = len(txx)
shxx = txx.shape
outvv = outV(txx)
neg_outvv = outV(neg_txx)
plt.plot(txx, outvv)
plt.plot(neg_txx, neg_outvv)
plt.show()
