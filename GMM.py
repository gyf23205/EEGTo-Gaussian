import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import glob
import os
from data import load_data
from sklearn.utils import shuffle

train_subject_files = glob.glob(os.path.join("D:/pycharm/Projects/EEGToGaussian/1D_data/train", "*.npz"))
X_train, Y_train, _, _ = load_data(train_subject_files, data_from_cluster=False)
means_init = np.zeros((5, 1))
for i in range(5):
    means_init[i, 0] = np.mean(X_train[i])
print(means_init)
X_train = np.concatenate((X_train[0][:500], X_train[1][:500], X_train[2][:500], X_train[3][:500], X_train[4][:500]), axis=0)
Y_train = np.concatenate((Y_train[0][:500], Y_train[1][:500], Y_train[2][:500], Y_train[3][:500], Y_train[4][:500]), axis=0)

test_subject_files = glob.glob(os.path.join("D:/pycharm/Projects/EEGToGaussian/1D_data/test", "*.npz"))
X_test, Y_test, _, _ = load_data(test_subject_files, data_from_cluster=False)
X_test = np.concatenate((X_test[0][:350], X_test[1][:350], X_test[2][:350], X_test[3][:350], X_test[4][:350]), axis=0)
Y_test = np.concatenate((Y_test[0][:350], Y_test[1][:350], Y_test[2][:350], Y_test[3][:350], Y_test[4][:350]), axis=0)
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

gm = GMM(n_components=5, random_state=0, means_init=means_init).fit(X_test)
print(gm.means_)
y_pred = gm.predict(X_test)
label_map = np.zeros((5, 5))
for i in range(y_pred.shape[0]):
    label_map[y_pred[i], Y_test[i]] += 1
print()


