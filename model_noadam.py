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
batch_num = np.ceil(X_train[0].shape[0]/batch_size)
train_db = tf.data.Dataset.from_tensor_slices(X_train[0]).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(X_train[0]).batch(1)
loss_hist = []
epochs = 500
lr = 0.01

# for step, x_train in enumerate(train_db):
#     print(step)
#     print(x_train)
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=100))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=100))
w2 = tf.Variable(tf.random.truncated_normal([3, 2], stddev=0.1, seed=233))
b2 = tf.Variable(tf.random.truncated_normal([2], stddev=0.1, seed=233))
w3 = tf.Variable(tf.random.truncated_normal([2, 1], stddev=0.1, seed=3))
b3 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1, seed=3))

# w1 = np.random.random([4, 3])

for epoch in range(epochs):
    loss_all = 0
    for step, x in enumerate(train_db):
    # for i in range(X_train[0].shape[0]):
    #     x = tf.constant([[X_train[0][i, :]]])
        with tf.GradientTape() as tape:
            y = tf.matmul(x, w1) + b1
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w2) + b2
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w3) + b3
            mean = tf.reduce_mean(y, axis=0)
            mu1 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 3.0), axis=0), batch_size)
            sigma1 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), batch_size), 1.5)
            mu2 = tf.divide(tf.reduce_sum(tf.pow(y - mean, 4.0), axis=0), batch_size)
            sigma2 = tf.pow(tf.divide(tf.reduce_sum(tf.pow(y - mean, 2.0), axis=0), batch_size), 2)
            # print('sigma1: {}'.format(sigma1))
            # print('sigma2: {}'.format(sigma2))
            S = tf.divide(mu1, sigma1)
            K = tf.divide(mu2, sigma2)
            loss = tf.reduce_sum(tf.multiply(batch_size/6, tf.multiply(tf.pow(S, 2.0), 10.0) \
                                             + tf.multiply(tf.pow(K - 3, 2.), 0.25)))
        if tf.math.is_nan(loss):
            break

        loss_all += loss
        grad = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grad[0])
        b1.assign_sub(lr * grad[1])
        w2.assign_sub(lr * grad[2])
        b2.assign_sub(lr * grad[3])
        w3.assign_sub(lr * grad[4])
        b3.assign_sub(lr * grad[5])
    if epoch == 200:
        lr = lr * 0.1
    if tf.math.is_nan(loss):
        break
    print('Epoch :{}, loss:{}'.format(epoch, loss_all/batch_num))

# np.save('net_weights3', np.array([w1.numpy(), b1.numpy(), w2.numpy(), b2.numpy(), w3.numpy(), b3.numpy()], dtype=object))


X_transformed = []
for x in test_db:
    y = tf.matmul(x, w1) + b1
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w2) + b2
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w3) + b3
    X_transformed.append(np.squeeze(y.numpy()))
X_transformed = np.array(X_transformed)
plt.hist(X_transformed)
plt.show()
print(skew(X_transformed))
print(kurtosis(X_transformed))
# visulize.histogram(X_transformed, 2)
print()


