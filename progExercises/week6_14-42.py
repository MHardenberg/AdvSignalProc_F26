#!/usr/bin/env python

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from matplotlib import rc

plt.switch_backend('qtagg')
plt.style.use('classic')

# Set LaTeX rendering and font settings
rc('text', usetex=True)
rc('font', family='serif', serif='Computer Modern', size=20)


def x(ns: np.ndarray) -> np.ndarray:
    phi1 = 2*np.pi*np.random.random(len(ns))
    phi2 = 2*np.pi*np.random.random(len(ns))
    wgn = np.random.normal(loc=0, scale=2)
    return np.cos(.44*np.pi * ns + phi1) + np.cos(.46*np.pi*ns+phi2) + wgn


def periodogram(xn: np.ndarray):
    X = np.fft.fft(xn)
    I = np.abs(X)
    I[1] = I[0]
    return I


xn = x(np.arange(1000))
X = periodogram(xn)


fig, ax = plt.subplots()
ax.semilogy(X)
ax.set_title('Random process realisation')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$S(\omega)$')


fig.savefig('figs/w6-42_periodograms.pdf')
plt.show()
