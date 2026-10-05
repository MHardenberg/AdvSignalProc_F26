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


def periodogram(xn: np.ndarray) -> np.ndarray:
    N = len(xn)
    X = np.fft.fft(xn)
    I = np.abs(X)[:N//2]
    I[1] = I[0]
    return I


def modperiodogram(xn: np.ndarray) -> np.ndarray:
    N = len(xn)
    w = sg.windows.hann(N)
    w /= np.linalg.norm(w)/np.sqrt(N)
    X = np.fft.fft(xn * w)
    I = np.abs(X)[:N//2]
    I[1] = I[0]
    return I


def ar2(ns: np.ndarray) -> np.ndarray:
    x = np.random.normal(loc=0, scale=1, size=len(ns))
    for i in range(1, len(ns)):
        x[i] += 3./4 * x[i-1]
        x[i] -= 1./2 * x[i-2] if i > 1 else 0
    return x


def ar4(ns: np.ndarray) -> np.ndarray:
    x = np.random.normal(loc=0, scale=1, size=len(ns))
    for i in range(1, len(ns)):
        x[i] += 2.7607 * x[i-1]
        x[i] -= 3.8106 * x[i-2] if i > 1 else 0
        x[i] += 2.6535 * x[i-3] if i > 2 else 0
        x[i] -= .9238 * x[i-4] if i > 3 else 0
    return x


fig, ax = plt.subplots(nrows=2, ncols=2)

ns = np.arange(1024)
ws = np.linspace(0, .5, len(ns)//2)
xar2 = ar2(ns)

Sar2 = periodogram(xar2)
Sar2mod = modperiodogram(xar2)
Sar2welch = sg.welch(xar2)

ax[0, 0].semilogy(ws, Sar2)
ax[0, 0].semilogy(Sar2welch[0], Sar2welch[1])

ax[0, 1].semilogy(ws, Sar2mod)
ax[0, 1].semilogy(Sar2welch[0], Sar2welch[1])

xar4 = ar4(ns)
Sar4 = periodogram(xar4)
Sar4mod = modperiodogram(xar4)
Sar4welch = sg.welch(xar4)

ax[1, 0].semilogy(ws, Sar4)
ax[1, 0].semilogy(Sar4welch[0], Sar4welch[1])

ax[1, 1].semilogy(ws, Sar4mod)
ax[1, 1].semilogy(Sar4welch[0], Sar4welch[1])


ax[0, 0].set_title('AR2 periodogram')
ax[0, 1].set_title('AR2 modified periodogram')
ax[1, 0].set_title('AR4 periodogram')
ax[1, 1].set_title('AR4 modified periodogram')
ax[1, 0].set_xlabel(r'$\omega$')
ax[1, 1].set_ylabel(r'$S(\omega)$')


fig.savefig('figs/w6-42_periodograms.pdf')
plt.show()
