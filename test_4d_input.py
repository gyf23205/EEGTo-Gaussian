import tensorflow as tf
import numpy as np
import visulize
import os
import data
from scipy.stats import skew, kurtosis, anderson
import matplotlib.pyplot as plt


X_test = data.load_unlabeled_data(['D:/pycharm/Projects/EEGToGaussian/features_per_class/train/feature_class4.npz'])
test_db = tf.data.Dataset.from_tensor_slices(X_test[0]).batch(1)
[w1, b1, w2, b2, w3, b3] = np.load('./model_4d_moreskew_leaky/more_skewness_net_weights1.npy', allow_pickle=True)
w1 = tf.convert_to_tensor(w1)
b1 = tf.convert_to_tensor(b1)
w2 = tf.convert_to_tensor(w2)
b2 = tf.convert_to_tensor(b2)
w3 = tf.convert_to_tensor(w3)
b3 = tf.convert_to_tensor(b3)
X_transformed = []
for x in test_db:
    y = tf.matmul(x, w1) + b1
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w2) + b2
    y = tf.nn.leaky_relu(y)
    y = tf.matmul(y, w3) + b3
    X_transformed.append(np.squeeze(y.numpy()))
X_transformed = np.array(X_transformed)
y = np.ones(shape=(X_transformed.shape[0])) * 4
# Save test features and labels
data_output_dir = './1D_data/train'
filename = f"{str(4).zfill(2)}.npz"
save_dict = {
    "x": X_transformed.astype(np.float32),
    "y": y.astype(np.int32),
    "fs": 100
}
np.savez(os.path.join(data_output_dir, filename), **save_dict)
plt.hist(X_transformed)
plt.show()
print(skew(X_transformed))
print(kurtosis(X_transformed))
[ad, _, _] = anderson(X_transformed)
print(ad)
# visulize.histogram(X_transformed, 2)
print()
