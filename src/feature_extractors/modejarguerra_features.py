#!/usr/bin/env python

"""
features_ECG.py
    
VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import numpy as np
import scipy.stats
import pywt
import operator
from numpy.polynomial.hermite import hermfit

# Compute the wavelet descriptor for a beat
def compute_wavelet_descriptor(beat, family='db1', level=3):
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]

# Compute the HOS descriptor for a beat
# Skewness (3 cumulant) and kurtosis (4 cumulant)
def compute_hos_descriptor(beat, n_intervals=11):
    lag = len(beat) / n_intervals
    hos_b = np.zeros(((n_intervals-1) * 3))
    for i in range(0, n_intervals-1):
        pose = (lag * (i+1))
        interval = beat[int(pose - (lag/2)):int(pose + (lag/2))]

        # Skewness
        hos_b[i] = scipy.stats.kstat(interval, 2)# scipy.stats.skew(interval, 0, True)
        if np.isnan(hos_b[i]):
            hos_b[i] = 0.0

        hos_b[(n_intervals-1) + i] = scipy.stats.kstat(interval, 3)
        if np.isnan(hos_b[(n_intervals-1) + i]):
            hos_b[(n_intervals-1) + i] = 0.0

        hos_b[(n_intervals-1)*2 + i] = scipy.stats.kstat(interval, 4)
        if np.isnan(hos_b[(n_intervals-1)*2 + i]):
            hos_b[(n_intervals-1)*2 + i] = 0.0

        # Kurtosis
        #hos_b[(n_intervals-1) + i] = scipy.stats.kurtosis(interval, 0, False, True)
    
    return hos_b


uniform_pattern_list = np.array([0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128,
                                 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255])
# Compute the uniform LBP 1D from signal with neigh equal to number of neighbours
# and return the 59 histogram:
# 0-57: uniform patterns
# 58: the non uniform pattern
# NOTE: this method only works with neigh = 8
def compute_Uniform_LBP(signal, neigh=8):
    hist_u_lbp = np.zeros(59, dtype=float)

    avg_win_size = 2
    # NOTE: Reduce sampling by half
    #signal_avg = scipy.signal.resample(signal, len(signal) / avg_win_size)

    for i in range(int(neigh/2), len(signal) - int(neigh/2)):
        pattern = np.zeros(neigh)
        ind = 0
        for n in [x for x in range(-1*int(neigh/2), 0)] + [x for x in range(1, int(neigh/2)+1)]:
            if signal[i] > signal[i+n]:
                pattern[ind] = 1
            ind += 1
        # Convert pattern to id-int 0-255 (for neigh == 8)
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        # Convert id to uniform LBP id 0-57 (uniform LBP)  58: (non uniform LBP)
        if pattern_id in uniform_pattern_list:
            pattern_uniform_id = int(np.argwhere(
                uniform_pattern_list == pattern_id))
        else:
            pattern_uniform_id = 58  # Non uniforms patternsuse

        hist_u_lbp[pattern_uniform_id] += 1.0

    return hist_u_lbp

# Compute my descriptor based on amplitudes of several intervals
def compute_my_own_descriptor(beat):
    L = len(beat)
    R_pos = int(len(beat) / 2)

    R_value = beat[R_pos]
    my_morph = np.zeros((4))
    y_values = np.zeros(4)
    x_values = np.zeros(4)
    # Obtain (max/min) values and index from the intervals
    [x_values[0], y_values[0]] = max(
        enumerate(beat[0:int(40*L/180)]), key=operator.itemgetter(1))
    [x_values[1], y_values[1]] = min(
        enumerate(beat[int(75*L/180):int(85*L/180)]), key=operator.itemgetter(1))
    [x_values[2], y_values[2]] = min(
        enumerate(beat[int(95*L/180):int(105*L/180)]), key=operator.itemgetter(1))
    [x_values[3], y_values[3]] = max(
        enumerate(beat[int(150*L/180):L]), key=operator.itemgetter(1))

    x_values[1] = x_values[1] + int(75*L/180)
    x_values[2] = x_values[2] + int(95*L/180)
    x_values[3] = x_values[3] + int(150*L/180)

    # Norm data before compute distance
    x_max = max(x_values)
    y_max = max(np.append(y_values, R_value))
    x_min = min(x_values)
    y_min = min(np.append(y_values, R_value))

    R_pos = (R_pos - x_min) / (x_max - x_min)
    R_value = (R_value - y_min) / (y_max - y_min)

    for n in range(0, 4):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        x_diff = (R_pos - x_values[n])
        y_diff = R_value - y_values[n]
        my_morph[n] = np.linalg.norm([x_diff, y_diff])
        # TODO test with np.sqrt(np.dot(x_diff, y_diff))

    if np.isnan(my_morph[n]):
        my_morph[n] = 0.0

    return my_morph

# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.polynomials.hermite.html
# Support Vector Machine-Based Expert System for Reliable Heartbeat Recognition
# 15 hermite coefficients!


def compute_HBF(beat_o):
    L = len(beat_o)
    x = range(0, L*2)
    beat = np.zeros(L*2)
    n = int(L / 2)
    m = np.mean([beat_o[0], beat_o[L-1]])
    for y in beat_o:
        beat[n] = y - m
        n += 1

    coeffs_hbf = np.zeros(15, dtype=float)
    coeffs_HBF_3 = hermfit(range(0, len(beat)), beat, 3)  # 3, 4, 5, 6?
    coeffs_HBF_4 = hermfit(range(0, len(beat)), beat, 4)
    coeffs_HBF_5 = hermfit(range(0, len(beat)), beat, 5)
    #coeffs_HBF_6 = hermfit(range(0,len(beat)), beat, 6)
    coeffs_HBF_20 = hermfit(range(0,len(beat)), beat, 20)

    coeffs_hbf = np.concatenate((coeffs_HBF_3, coeffs_HBF_4, coeffs_HBF_5, coeffs_HBF_20))

    return coeffs_HBF_20[1:]
