import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import glob
import os
from data import load_data
from sklearn.utils import shuffle
import tensorflow as tf


# def get_1D_feature(X_test, train_pdf):
#     # Build 5 network and feed in band power data3
#     temp_results = np.zeros((X_test.shape[0], 5))
#     results = np.zeros(X_test.shape[0])
#     c = np.zeros(X_test.shape[0])
#     for i in range(5):
#         [w1, b1, w2, b2, w3, b3] = np.load('./model_4d_moreskew_leaky/more_skewness_net_weights{}.npy'.format(i), allow_pickle=True)
#         w1 = tf.convert_to_tensor(w1)
#         b1 = tf.convert_to_tensor(b1)
#         w2 = tf.convert_to_tensor(w2)
#         b2 = tf.convert_to_tensor(b2)
#         w3 = tf.convert_to_tensor(w3)
#         b3 = tf.convert_to_tensor(b3)
#         test_db = tf.data.Dataset.from_tensor_slices(X_test).batch(1)
#         for step, x in enumerate(test_db):
#             y = tf.matmul(x, w1) + b1
#             y = tf.nn.leaky_relu(y)
#             y = tf.matmul(y, w2) + b2
#             y = tf.nn.leaky_relu(y)
#             y = tf.matmul(y, w3) + b3
#             temp_results[step, i] = np.squeeze(y.numpy())
#     for i in range(temp_results.shape[0]):
#         counts = np.zeros(5)
#         for j in range(temp_results.shape[1]):
#             upper = temp_results[i, j] + 80
#             down = temp_results[i, j] - 80
#             counts[j] = np.count_nonzero((down < train_pdf[j, :]) & (train_pdf[j, :] < upper)) * (np.max(train_pdf[j, :]) - np.min(train_pdf[j, :]))
#         results[i] = temp_results[i, np.where(counts == np.max(counts))[0]]
#         c[i] = np.where(counts == np.max(counts))[0]
#     return results, c

def get_1D_feature(X_test, train_pdf):
    # Build 5 network and feed in band power data3
    temp_results = np.zeros((X_test.shape[0], 5))
    results = np.zeros(X_test.shape[0])
    c = np.zeros(X_test.shape[0])
    for i in range(5):
        [w1, b1, w2, b2, w3, b3] = np.load('./model_4d_moreskew_leaky/more_skewness_net_weights{}.npy'.format(i), allow_pickle=True)
        w1 = tf.convert_to_tensor(w1)
        b1 = tf.convert_to_tensor(b1)
        w2 = tf.convert_to_tensor(w2)
        b2 = tf.convert_to_tensor(b2)
        w3 = tf.convert_to_tensor(w3)
        b3 = tf.convert_to_tensor(b3)
        test_db = tf.data.Dataset.from_tensor_slices(X_test).batch(1)
        for step, x in enumerate(test_db):
            y = tf.matmul(x, w1) + b1
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w2) + b2
            y = tf.nn.leaky_relu(y)
            y = tf.matmul(y, w3) + b3
            temp_results[step, i] = np.squeeze(y.numpy())
    from scipy.stats import norm
    means = np.mean(train_pdf, axis=1)
    std = np.std(train_pdf, axis=1)
    for i in range(temp_results.shape[0]):
        counts = np.zeros(5)
        for j in range(temp_results.shape[1]):
            counts[j] = norm.pdf(temp_results[i, j], means[j], std[j])
        results[i] = temp_results[i, np.where(counts == np.max(counts))[0]]
        c[i] = np.where(counts == np.max(counts))[0]
    return results, c



train_subject_files = glob.glob(os.path.join("D:/pycharm/Projects/EEGToGaussian/1D_data/train", "*.npz"))
X_train, Y_train, _, _ = load_data(train_subject_files, data_from_cluster=False)
naive_train_pdf = np.array([X_train[0][:500], X_train[1][:500], X_train[2][:500], X_train[3][:500], X_train[4][:500]])
X_train = np.concatenate((X_train[0][:500], X_train[1][:500], X_train[2][:500], X_train[3][:500], X_train[4][:500]), axis=0)
Y_train = np.concatenate((Y_train[0][:500], Y_train[1][:500], Y_train[2][:500], Y_train[3][:500], Y_train[4][:500]), axis=0)
test_subject_files = glob.glob(os.path.join("./band_power_features/test", "*.npz"))
X_test, Y_test, _, _ = load_data(test_subject_files, data_from_cluster=False)
X_test = np.concatenate((X_test[0], X_test[1], X_test[2], X_test[3], X_test[4]), axis=0)
Y_test = np.concatenate((Y_test[0], Y_test[1], Y_test[2], Y_test[3], Y_test[4]), axis=0)
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
X_test, c = get_1D_feature(X_test, naive_train_pdf)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
clf = LDA()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
acc = np.sum(Y_pred == Y_test)/Y_pred.shape[0]
recall = np.zeros(5)
for i in range(5):
    idx = np.where(Y_test == i)[0]
    recall[i] = np.where(Y_pred[idx] == i)[0].shape[0]/len(idx)
recall = np.mean(recall)
print(acc)
print(recall)

