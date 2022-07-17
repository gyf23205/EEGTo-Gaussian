import matplotlib.pyplot as plt
import numpy as np


def histogram(x, band_num):
    col_num = int(band_num/2)
    for i in range(band_num):
        plt.subplot(2, col_num, i+1)
        plt.hist(x[:, i])
    plt.show()
