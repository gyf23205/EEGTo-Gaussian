import tensorflow as tf
import numpy as np
import visulize
import glob
import os
import data
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

'''
Model trained with lr decreasing. Still treat 4 input in the same way.
'''

# subject_files = glob.glob(os.path.join("./features_per_class", "*.npz"))
X_train = data.load_unlabeled_data(['D:/pycharm/Projects/EEGToGaussian/features_per_class/train/feature_class1.npz'])
# mean = np.mean(X_train[0], axis=0)
# var = np.var(X_train[0], axis=0)
# X_train[0] = (X_train[0]-mean)/np.sqrt(var)
# visulize.histogram(X_train[0], 4)
batch_size = 128
val_freq = 50
X_train = X_train[0]
X_train = tf.random.shuffle(X_train, seed=4)
val_ratio = 0.2
split = int(X_train.shape[0] * (1 - val_ratio))
batch_num = np.ceil(X_train.shape[0] * (1 - val_ratio) / batch_size)
batch_num_val = np.ceil(X_train.shape[0] * val_ratio / batch_size)
train_db = tf.data.Dataset.from_tensor_slices(X_train[:split, :]).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices(X_train[split:, :]).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(X_train).batch(1)
loss_hist = []
epochs = 500
lr = 0.01
beta = 0.9
epsilon = 10 ** -6
# for step, x_train in enumerate(train_db):
#     print(step)
#     print(x_train)
w1 = tf.Variable(tf.random.truncated_normal([4, 10], stddev=0.1, seed=100))
b1 = tf.Variable(tf.random.truncated_normal([10], stddev=0.1, seed=100))
w2 = tf.Variable(tf.random.truncated_normal([10, 5], stddev=0.1, seed=233))
b2 = tf.Variable(tf.random.truncated_normal([5], stddev=0.1, seed=233))
w3 = tf.Variable(tf.random.truncated_normal([5, 1], stddev=0.1, seed=3))
b3 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1, seed=3))
tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)

# w1 = np.random.random([4, 3])
early_stop_count = 0
loss_min = np.inf
for epoch in range(epochs):
    r = [tf.zeros(shape=w1.shape), tf.zeros(shape=b1.shape), tf.zeros(shape=w2.shape),\
         tf.zeros(shape=b2.shape), tf.zeros(shape=w3.shape), tf.zeros(shape=b3.shape)]
    loss_all = 0
    for step, x in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x, w1) + b1
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w2) + b2
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w3) + b3
            y = tf.nn.sigmoid(y)
            sort = np.argsort(y.numpy(), axis=0)
            idx = np.ones(y.shape[0])
            for s, i in enumerate(sort):
                idx[i] = s
            idx = tf.constant(idx + 1, dtype=float)
            mean = tf.reduce_mean(y, axis=0)
            var = tf.math.reduce_variance(y, axis=0)
            y = (y - mean) / tf.sqrt(var)
            loss = -batch_size - (1. / batch_size) * \
                   tf.reduce_sum(tf.expand_dims(2. * idx - 1., axis=1) * tf.math.log(dist.cdf(y)) \
                                 + tf.expand_dims(2. * (batch_size - idx) + 1., axis=1) * tf.math.log(1. - dist.cdf(y)))
            if tf.math.is_nan(loss):
                break

        loss_all += loss
        grad = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        delta = []
        for k in range(len(r)):
            r[k] = r[k] * beta + (1 - beta) * tf.pow(grad[k], 2)
            delta.append(-lr / tf.sqrt(epsilon + r[k]) * grad[k])

        w1.assign_sub(delta[0])
        b1.assign_sub(delta[1])
        w2.assign_sub(delta[2])
        b2.assign_sub(delta[3])
        w3.assign_sub(delta[4])
        b3.assign_sub(delta[5])
    if epoch == 200:  # tune learning rate
        lr = lr * 0.1
    if epoch % val_freq == 0:  # validation
        early_stop_count += 1
        loss_val_all = 0
        for step, x in enumerate(val_db):
            with tf.GradientTape() as tape:
                y = tf.matmul(x, w1) + b1
                y = tf.nn.leaky_relu(y)
                y = tf.matmul(y, w2) + b2
                y = tf.nn.leaky_relu(y)
                y = tf.matmul(y, w3) + b3
                y = tf.nn.sigmoid(y)
                sort = np.argsort(y.numpy(), axis=0)
                idx = np.ones(y.shape[0])
                for s, i in enumerate(sort):
                    idx[i] = s
                idx = tf.constant(idx + 1, dtype=float)
                mean = tf.reduce_mean(y, axis=0)
                var = tf.math.reduce_variance(y, axis=0)
                y = (y - mean) / tf.sqrt(var)
                loss_val = -batch_size - (1. / batch_size) * \
                           tf.reduce_sum(tf.expand_dims(2. * idx - 1., axis=1) * tf.math.log(dist.cdf(y)) \
                                         + tf.expand_dims(2. * (batch_size - idx) + 1., axis=1) * tf.math.log(
                               1. - dist.cdf(y)))
                loss_val_all += loss_val
        print('-------Epoch :{}, loss_val:{}--------'.format(epoch, loss_val_all / batch_num_val))
    if loss_min > loss.numpy():
        loss_min = loss.numpy()
        early_stop_count = 0
    if early_stop_count > 1:
        break
    print('Epoch :{}, loss:{}'.format(epoch, loss_all / batch_num))
    if tf.math.is_nan(loss):  # stop when nan
        break

# np.save('net_weights4', np.array([w1.numpy(), b1.numpy(), w2.numpy(), b2.numpy(), w3.numpy(), b3.numpy()], dtype=object))

X_transformed = []
for x in test_db:
    y = tf.matmul(x, w1) + b1
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w2) + b2
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w3) + b3
    y = tf.nn.sigmoid(y)
    X_transformed.append(np.squeeze(y.numpy()))
X_transformed = np.array(X_transformed)
plt.hist(X_transformed)
plt.show()
print(skew(X_transformed))
print(kurtosis(X_transformed))
# visulize.histogram(X_transformed, 2)
print()
