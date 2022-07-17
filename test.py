import tensorflow as tf
import numpy as np
import visulize
import data
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


X_test = data.load_unlabeled_data(['D:/pycharm/Projects/EEGToGaussian/features_per_class/train/feature_class4.npz'])
test_db = tf.data.Dataset.from_tensor_slices(X_test[0]).batch(1)
[w1, b1, w2, b2, w3, b3] = np.load('partical_connnect_net_weights4.npy', allow_pickle=True)
w1 = tf.convert_to_tensor(w1)
b1 = tf.convert_to_tensor(b1)
w2 = tf.convert_to_tensor(w2)
b2 = tf.convert_to_tensor(b2)
w3 = tf.convert_to_tensor(w3)
b3 = tf.convert_to_tensor(b3)
X_transformed = []
for x in test_db:
    y1 = tf.matmul(x[:, :2], w1) + b1
    y1 = tf.nn.relu(y1)
    y2 = tf.matmul(x[:, 2:], w2) + b2
    y2 = tf.nn.relu(y2)
    y = tf.concat([y1, y2], axis=1)
    y = tf.matmul(y, w3) + b3
    X_transformed.append(np.squeeze(y.numpy()))
X_transformed = np.array(X_transformed)
plt.hist(X_transformed)
plt.show()
print(skew(X_transformed))
print(kurtosis(X_transformed))
# visulize.histogram(X_transformed, 2)
print()
