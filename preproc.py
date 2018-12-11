# MIT License
#
# Copyright (c) 2018 Pedro Virgilio Basilio Jeronymo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
SVHN Dataset Pre-processor

- Pre-processes SVHN dataset to a more convinient format for my ML models.
- Provides some utilities as flatten.
- Run it as main to pre-process the data.

More info on github.com/pedrovbj/SVHN-ML
'''

import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle

# Path to data directory
path = '../SVHN/'

def y2indicator(y, n_classes=10):
    '''
    Transforms a categorical targets vector to a one-hot encoded vector

    y: input vector
    n_classes: number of classes

    Example:
        INPUT:  y = np.array([0, 1, 2]), n_classes=3
        OUTPUT: y_ind = np.array([[1, 0, 0]
                                  [0, 1, 0]
                                  [0, 0, 1]])
    '''
    N = len(y)
    ind = np.zeros((N, n_classes))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def flatten(X):
    '''
    Flatten all but the first dimension of a numpy array X
    '''
    return X.reshape(X.shape[0], -1)

def rearrange(X):
    '''
    Rearrange the input array X
    from (x-coord, y-coord, features, sample number)
    to (sample number, features, x-coord, y-coord)
    '''
    N = X.shape[-1]
    out = np.zeros((N, 3, 32, 32), dtype=np.float32)

    for i in range(N):
        for j in range(3):
            out[i, j, :, :] = X[:, :, j, i]
    return out

def load_data():
    '''
    Loads the previously pre-processed data
    '''
    Xtrain = np.load(path+'Xtrain.npy')
    Ytrain = np.load(path+'Ytrain.npy')
    Ytrain_ind = np.load(path+'Ytrain_ind.npy')

    Xtest = np.load(path+'Xtest.npy')
    Ytest = np.load(path+'Ytest.npy')
    Ytest_ind = np.load(path+'Ytest_ind.npy')

    return Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind

if __name__ == "__main__":
    train = loadmat(path+'train_32x32.mat')
    test = loadmat(path+'test_32x32.mat')

    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten()-1
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten()-1
    Xtest, Ytest = shuffle(Xtest, Ytest)
    del test
    Ytest_ind = y2indicator(Ytest)

    np.save(path+'Xtrain', Xtrain)
    np.save(path+'Ytrain', Ytrain)
    np.save(path+'Ytrain_ind', Ytrain_ind)
    np.save(path+'Xtest', Xtest)
    np.save(path+'Ytest', Ytest)
    np.save(path+'Ytest_ind', Ytest_ind)
