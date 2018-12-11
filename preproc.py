import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle

# Path to data directory
path = '../SVHN/'

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def flatten(X):
    return X.reshape(X.shape[0], -1)

def rearrange(X):
    N = X.shape[-1]
    out = np.zeros((N, 3, 32, 32), dtype=np.float32)

    for i in range(N):
        for j in range(3):
            out[i, j, :, :] = X[:, :, j, i]
    return out/255

def load_data():
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
