# Python modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# My modules
from preproc import load_data, flatten
from mylogger import MyLogger

# Init logger
prefix = 'ANN'
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = 'results/{}_out_{}.txt'.format(prefix, timestamp)
print(log_file, end='\n')
logger = MyLogger(log_file)

## Disable TF log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Load data
logger.debug('Loading data... ')
t0 = datetime.now()
Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind = load_data()
Xtrain = flatten(Xtrain)
Xtest = flatten(Xtest)
Xtrain = (Xtrain-Xtrain.mean())/Xtrain.std()
Xtest = (Xtest-Xtest.mean())/Xtest.std()
dt = datetime.now()-t0
logger.debug('Done. [Elapsed {}]\r\n'.format(dt))

## Define and fit model
logger.debug('Model fitting...\r\n')
t0 = datetime.now()

# logger.debug period for cost of test set and accuracy
logger.debug_period = 50

# Number of samples to take from test set each time it computes cost
n_samples = 5000

# Hyperparameters
lr = 1e-6 # learning rate
max_iter = 200 # maximum number of epochs
n_batches = 100 # number of training batches
batch_size = Xtrain.shape[0]//n_batches # training batch size
D = Xtrain.shape[1] # input layer size
M1 = 1000           # 1st dense layer size
M2 = 500            # 2nd dense layer size
K = 10              # output layer size

# Weights and biases initialization
W1_init = np.random.randn(D, M1) / np.sqrt(D+M1)
b1_init = np.zeros(M1)
W2_init = np.random.randn(M1, M2) / np.sqrt(M1+M2)
b2_init = np.zeros(M2)
W3_init = np.random.randn(M2, K) / np.sqrt(M2+K)
b3_init = np.zeros(K)

# Input and Output placeholders
X = tf.placeholder(name='X', dtype=np.float32)
T = tf.placeholder(name='T', dtype=np.float32)

# TF Weights and Biases variables
W1 = tf.Variable(W1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
W2 = tf.Variable(W2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))
W3 = tf.Variable(W3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))

# Forwarding
Z1 = tf.nn.relu(tf.matmul(X, W1)+b1)
Z2 = tf.nn.relu(tf.matmul(Z1, W2)+b2)
Yish = tf.matmul(Z2, W3)+b3

# Cost function (cross entropy)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2 \
    (logits=Yish, labels=T))

# Train operation
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# Prediction operation
predict_op = tf.argmax(Yish, 1)

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
            Tb = Ytrain_ind[j*batch_size:(j+1)*batch_size,]

            # Training
            session.run(train_op, feed_dict={X: Xb, T: Tb})

            # Cost logger.debuging
            if j % logger.debug_period == 0:
                idx_samples = np.random.choice(Xtest.shape[0], n_samples, \
                    replace=False)
                test_cost = session.run(cost, \
                    feed_dict={X: Xtest[idx_samples],T: Ytest_ind[idx_samples]})
                Ypred = session.run(predict_op, \
                    feed_dict={X: Xtest[idx_samples],T: Ytest_ind[idx_samples]})
                acc = 100*np.mean(Ypred == Ytest[idx_samples])
                dt = datetime.now()-t0
                est_t = (max_iter*n_batches)*dt/(i*n_batches+j)
                logger.debug('E: {:03d}/{} B: {:03d}/{} C: {:.6f} A: {:.2f}% [Elapsed {} of ~ {}]\r\n'\
                    .format(i, max_iter, j, n_batches, test_cost, acc, dt, est_t))
                costs.append(test_cost)

    # Save session
    save_path = saver.save(session, 'results/{}_model_{}.ckpt' \
        .format(prefix, timestamp))
    logger.debug('Model saved in path: {}'.format(save_path))

dt = datetime.now()-t0
logger.debug('Done. [Elapsed {}]\r\n'.format(dt))

# Shutdown logger
logger.shutdown()

## Plots cross entropy and saves it to disk
plt.plot(logger.debug_period*np.arange(1, len(costs)+1), costs, marker='x')
plt.title('[{}] Cross entropy for {} samples from test set\nat every {} weight updates'\
    .format(prefix, n_samples, logger.debug_period))
plt.xlabel('Number of weight updates')
plt.ylabel('Cross entropy')
plt.savefig('results/{}_cross_entropy_{}.png'.format(prefix, timestamp))
plt.show()
