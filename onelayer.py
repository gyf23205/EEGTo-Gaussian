import tensorflow as tf
import numpy as np
import visulize
import glob
import os
import data
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

'''
Model trained with lr decreasing. Still treat 4 input in the same way.
'''

# subject_files = glob.glob(os.path.join("./features_per_class", "*.npz"))
X_train = data.load_unlabeled_data(['D:/pycharm/Projects/EEGToGaussian/features_per_class/train/feature_class1.npz'])
# mean = np.mean(X_train[0], axis=0)
# var = np.var(X_train[0], axis=0)
# X_train[0] = (X_train[0]-mean)/np.sqrt(var)
# visulize.histogram(X_train[0], 4)
batch_size = 64
val_freq = 50
X_train = X_train[0]
X_train = tf.random.shuffle(X_train, seed=4)
val_ratio = 0.2
split = int(X_train.shape[0] * (1 - val_ratio))
batch_num = np.ceil(X_train.shape[0] * (1 - val_ratio)/batch_size)
batch_num_val = np.ceil(X_train.shape[0] * val_ratio/batch_size)
train_db = tf.data.Dataset.from_tensor_slices(X_train[:split, :]).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices(X_train[split:, :]).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(X_train).batch(1)
loss_hist = []
epochs = 500
lr = 0.01
# for step, x_train in enumerate(train_db):
#     print(step)
#     print(x_train)
w1 = tf.Variable(tf.random.truncated_normal([4, 30], stddev=0.1, seed=2))
b1 = tf.Variable(tf.random.truncated_normal([30], stddev=0.1, seed=23))
w2 = tf.Variable(tf.random.truncated_normal([30, 1], stddev=0.1, seed=3))
b2 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1, seed=3))

# w1 = np.random.random([4, 3])
early_stop_count = 0
loss_min = np.inf
for epoch in range(epochs):
    loss_all = 0
    for step, x in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x, w1) + b1
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w2) + b2
            mean = tf.reduce_mean(y, axis=0)
            mu1 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 3.0), axis=0), batch_size)
            sigma1 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), batch_size), 1.5)
            mu2 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 4.0), axis=0), batch_size)
            sigma2 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), batch_size), 2)
            # print('sigma1: {}'.format(sigma1))
            # print('sigma2: {}'.format(sigma2))
            S = tf.divide(mu1, sigma1)
            K = tf.divide(mu2, sigma2)
            loss = tf.reduce_sum(tf.multiply(batch_size/6, tf.multiply(tf.pow(S, 2.0), 10.0) + tf.multiply(tf.pow(K - 3, 2.), 0.25)))
        if tf.math.is_nan(loss):
            break

        loss_all += loss
        grad = tape.gradient(loss, [w1, b1, w2, b2])
        w1.assign_sub(lr * grad[0])
        b1.assign_sub(lr * grad[1])
        w2.assign_sub(lr * grad[2])
        b2.assign_sub(lr * grad[3])
    if epoch == 200:  # tune learning rate
        lr = lr * 0.1
    if tf.math.is_nan(loss):  # stop when nan
        break
    if epoch % val_freq == 0:  # validation
        early_stop_count += 1
        loss_val_all = 0
        for step, x in enumerate(val_db):
            with tf.GradientTape() as tape:
                y = tf.matmul(x, w1) + b1
                y = tf.nn.leaky_relu(y)
                y = tf.matmul(y, w2) + b2
                mean = tf.reduce_mean(y, axis=0)
                mu1 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 3.0), axis=0), batch_size)
                sigma1 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), batch_size), 1.5)
                mu2 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 4.0), axis=0), batch_size)
                sigma2 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), batch_size), 2)
                # print('sigma1: {}'.format(sigma1))
                # print('sigma2: {}'.format(sigma2))
                S = tf.divide(mu1, sigma1)
                K = tf.divide(mu2, sigma2)
                loss_val = tf.reduce_sum(tf.multiply(batch_size / 6,
                                                 tf.multiply(tf.pow(S, 2.0), 10.0) + tf.multiply(tf.pow(K - 3, 2.),
                                                                                                0.25)))
                loss_val_all += loss_val
        print('-------Epoch :{}, loss_val:{}--------'.format(epoch, loss_val_all / batch_num_val))
    if loss_min > loss.numpy():
        loss_min = loss.numpy()
        early_stop_count = 0
    if early_stop_count > 1:
        break
    print('Epoch :{}, loss:{}'.format(epoch, loss_all/batch_num))

np.save('net_weights1', np.array([w1.numpy(), b1.numpy(), w2.numpy(), b2.numpy()], dtype=object))


X_transformed = []
for x in test_db:
    y = tf.matmul(x, w1) + b1
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w2) + b2
    X_transformed.append(np.squeeze(y.numpy()))
X_transformed = np.array(X_transformed)
plt.hist(X_transformed)
plt.show()
print(skew(X_transformed))
print(kurtosis(X_transformed))
# visulize.histogram(X_transformed, 2)
print()


