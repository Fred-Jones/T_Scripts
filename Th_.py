import pandas as pd
import numpy as np

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

print y_train.ix[:,0]
