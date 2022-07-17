import math
from scipy.signal import windows
from scipy.fft import fft, fftfreq
import numpy as np


def feature_extractor(signal, Fs):
    window_len = 30
    M = window_len*Fs
    freq_resolution = 0.5
    NW = window_len*freq_resolution/2
    Kmax = math.floor(2 * NW)
    tapers = windows.dpss(M, NW, Kmax)
    signal = np.tile(signal, (Kmax, 1))
    signal_taped = np.multiply(signal, tapers)
    fft_data = np.zeros((Kmax, M))
    for i in range(Kmax):
        fft_data[i, :] = np.abs(fft(signal_taped[i, :]))
    xf = fftfreq(M, 1/Fs)[:M // 2]
    power = np.mean(2.0 / M * fft_data[:, 0:M // 2], 0)

    beta_range = np.array([12.5, 25])
    alpha_range = np.array([7.5, 12.5])
    theta_range = np.array([3.5, 7.5])
    delta_range = np.array([1.5, 3.5])

    freq_step = 1/window_len

    beta_index = np.floor(beta_range / freq_step).astype(int)
    alpha_index = np.floor(alpha_range / freq_step).astype(int)
    theta_index = np.floor(theta_range / freq_step).astype(int)
    delta_index = np.floor(delta_range / freq_step).astype(int)
    band_power = [np.sum(power[beta_index[0]:beta_index[1]]), np.sum(power[alpha_index[0]:alpha_index[1]]),
                  np.sum(power[theta_index[0]:theta_index[1]]), np.sum(power[delta_index[0]:delta_index[1]])]
    return band_power