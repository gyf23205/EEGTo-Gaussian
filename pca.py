import data
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import skew, kurtosis, anderson
import matplotlib.pyplot as plt


X_train = data.load_unlabeled_data(['D:/pycharm/Projects/EEGToGaussian/features_per_class/test/feature_class4.npz'])
X = np.zeros(shape=X_train[0].shape)
for i in range(X_train[0].shape[0]):
    X[i] = X_train[0][i]/np.sum(X_train[0][i])
X = PCA(n_components=1).fit_transform(X_train[0])
X = np.squeeze(X)
# X = [lambda x: np.log(x/1-x) for x in X]
plt.hist(X)
plt.show()
print(skew(X))
print(kurtosis(X))
[ad, _, _] = anderson(X)
print(ad)
print()