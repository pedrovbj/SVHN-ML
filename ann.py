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
Artifical Neural Network Model

ANN model for SVHN Dataset
More info on github.com/pedrovbj/SVHN-ML
'''

# Python modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# My modules
from preproc import load_data, flatten
from mylogger import MyLogger

# Paths
prefix = 'ANN'
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_path = Path('{}_results'.format(prefix))
log_path = results_path / '{}_out_{}.txt'.format(prefix, timestamp)
model_path = results_path / '{}_model_{}.ckpt'.format(prefix, timestamp)
img_path = results_path / '{}_cross_entropy_{}.png'.format(prefix, timestamp)

# Init logger
print(log_path, end='\r\n')
logger = MyLogger(log_path)

## Disable TF log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Load data
logger.debug('Loading data... ')
t0 = datetime.now()
Xtrain, Ytrain, Xtest, Ytest = load_data()
Xtrain = flatten(Xtrain)
Xtest = flatten(Xtest)
Xtrain = (Xtrain-Xtrain.mean())/Xtrain.std()
Xtest = (Xtest-Xtest.mean())/Xtest.std()
dt = datetime.now()-t0
logger.debug('Done. [Elapsed {}]\r\n'.format(dt))

## Define and fit model
logger.debug('Model fitting...\r\n')
t0 = datetime.now()

# printing period for cost of test set and accuracy
print_period = 1

# Number of samples to take from test set each time it computes cost
# n_samples = 10000

# Hyperparameters
lr = 1e-4 # learning rate
max_iter = 500 # maximum number of epochs
n_batches = 4 # number of training batches
batch_size = Xtrain.shape[0]//n_batches # training batch size
D = Xtrain.shape[1]
K = 10
n_layers = 3
layer_sizes = [D, *map(lambda k: D*(2**k)//(3**k)+K, range(1, n_layers)), K]

def init_layer(k, layer_sizes):
    M0 = layer_sizes[k-1]
    M1 = layer_sizes[k]
    return np.random.randn(M0, M1)*2/np.sqrt(M0+M1)
def init_bias(k, layer_sizes):
    return np.zeros(layer_sizes[k])

# Weights and biases initialization
W_init = [init_layer(k, layer_sizes) for k in range(1, n_layers+1)]
b_init = [init_bias(k, layer_sizes) for k in range(1, n_layers+1)]

# Input and Output placeholders
inputs = tf.placeholder(name='inputs', dtype=np.float32)
labels = tf.placeholder(name='labels', dtype=np.int32)

# TF Weights and Biases variables
W = [tf.Variable(W_init[k], dtype=np.float32) for k in range(n_layers)]
b = [tf.Variable(b_init[k], dtype=np.float32) for k in range(n_layers)]

# Forwarding
def forward_train(X, dropout_rates):
    Z = X
    for k in range(n_layers-1):
        Z = tf.nn.relu(tf.matmul(Z, W[k])+b[k])
        Z = tf.nn.dropout(Z, dropout_rates[k])
    return tf.matmul(Z, W[-1])+b[-1]

def forward_test(X):
    Z = X
    for k in range(n_layers-1):
        Z = tf.nn.relu(tf.matmul(Z, W[k])+b[k])
    return tf.matmul(Z, W[-1])+b[-1]

train_logits = forward_train(inputs, [0.5]*n_layers)
test_logits = forward_test(inputs)

# Cost function (cross entropy)
train_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits \
    (logits=train_logits, labels=labels))
test_cost =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits \
    (logits=test_logits, labels=labels))

# Train operation
train_op = tf.train.AdamOptimizer(lr).minimize(train_cost)

# Prediction operation
predict_op = tf.argmax(forward_test(inputs), 1)

# Add operation to save session
saver = tf.train.Saver()

# TF session
costs = [] # Cross entropy of test set samples
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1, max_iter+1):
        for j in range(1, n_batches+1):
            # Select batch
            Xb = Xtrain[j*batch_size:(j+1)*batch_size,]
            Tb = Ytrain[j*batch_size:(j+1)*batch_size,]

            # Training
            session.run(train_op, feed_dict={inputs: Xb, labels: Tb})

            # Cost printing
            if j % print_period == 0:
                #idx_samples = np.random.choice(Xtest.shape[0], n_samples, \
                #    replace=False)
                #Xs = Xtest[idx_samples]
                #Ys = Ytest[idx_samples]
                Xs = Xtest
                Ys = Ytest
                tc = session.run(test_cost, \
                    feed_dict={inputs: Xs, labels: Ys})
                Ypred = session.run(predict_op, \
                    feed_dict={inputs: Xs, labels: Ys})
                acc = 100*np.mean(Ypred == Ys)
                dt = datetime.now()-t0
                est_t = (max_iter*n_batches)*dt/(i*n_batches+j)
                logger.debug('E: {:03d}/{} B: {:03d}/{} C: {:.6f} A: {:.2f}% [Elapsed {} of ~ {}]\r\n'\
                    .format(i, max_iter, j, n_batches, tc, acc, dt, est_t))
                costs.append(tc)

    # Save session
    saver.save(session, str(model_path))
    logger.debug('Model saved in path: {}\r\n'.format(model_path))

dt = datetime.now()-t0
logger.debug('Done. [Elapsed {}]\r\n'.format(dt))

# Shutdown logger
logger.shutdown()

## Plots cross entropy and saves it to disk
plt.plot(print_period*np.arange(1, len(costs)+1), costs, marker='x')
plt.title('[{}] Cross entropy for test set\nat every {} weight updates'\
    .format(prefix, print_period))
plt.xlabel('Number of weight updates')
plt.ylabel('Cross entropy')
plt.savefig(img_path)
plt.show()
