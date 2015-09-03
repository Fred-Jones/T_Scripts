###
#Running MNIST data through example from deeplearning.net
###


import numpy as np
import pandas as pd
from theano import *
import theano.tensor as T

def getData(train='/train.csv', test='/test.csv', path='/Users/sethmartinez/desktop/Kaggle/DigitRecog'):
    _path = path

    with open(_path + train, 'rb') as train_file:
        train_data = pd.read_csv(train_file, sep=',')

    with open(_path + test, 'rb') as test_file:
        test_data = pd.read_csv(test_file, sep=',')
    return train_data, test_data

##Set up data
xy_train, xy_test = getData()
x_train = xy_train.ix[:,1:]
y_train = xy_train[['label']]
x_test = xy_test
nfeatures = len(x_train)

##Set up theano
w = theano.shared(np.random.randn(784), name='w')
X = T.dmatrix('X')
y = T.dvector('y')
b = theano.shared(0., name='b')

z = 1/(1+T.exp(-T.dot(X, w)-b))
prediction = z > 0.5
xent = -y * T.log(z) - (1-y) * T.log(1-z)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])

train = theano.function(
                inputs = [X,y],
                outputs = [prediction, xent],
                updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[X], outputs=prediction)

##Train
steps = 10000
print x_train.shape
print 'Training...'
for i in range(steps):
    if i%20 == 0:
        print '20 iterations'
    pred, err = train(x_train, y_train.ix[:,0])
print w.get_value()
print predict(x_test)
