#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:27:53 2020

Fourier Analysis of waves using numpy fft 

@author: simo94
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# # Number of samplepoints
N = 1000
# # sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = 10 + np.sin(80.0 * 2.0*np.pi*x) + 0.5*np.sin(5.0 * 2.0*np.pi*x)
# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# plt.subplot(2, 1, 1)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
# plt.subplot(2, 1, 2)
# plt.plot(xf[1:], 2.0/N * np.abs(yf[0:N/2])[1:])
plt.plot(y)

Y  = np.fft.fft(y)
freq = np.fft.fftfreq(len(y),T)

plt.figure()
plt.plot(freq[1:np.int(N/2)], np.abs(Y)[1:np.int(N/2)])

idx = find_peaks(np.abs(Y)[1:np.int(N/2)])
print('maximum angular frequency for w: ',freq[1+idx[0]])     