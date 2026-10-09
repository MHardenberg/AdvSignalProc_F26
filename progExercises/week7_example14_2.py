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

# Sar2 = periodogram(xar2)
# Sar2mod = modperiodogram(xar2)
pws, Sar2 = sg.periodogram(xar2)
_, Sar2mod = sg.periodogram(xar2, window='hann')
wws, Sar2welch = sg.welch(xar2)

ax[0, 0].semilogy(pws[1:], Sar2[1:], label='periodogram')
ax[0, 0].semilogy(wws, Sar2welch, lw=2, label='Welch estimator')

ax[0, 1].semilogy(pws[1:], Sar2mod[1:], label='modified periodogram')
ax[0, 1].semilogy(wws, Sar2welch, lw=2, label='Welch estimator')

xar4 = ar4(ns)
# Sar4 = periodogram(xar4)
# Sar4mod = modperiodogram(xar4)

pws, Sar4 = sg.periodogram(xar4)
_, Sar4mod = sg.periodogram(xar4, window='hann')
wws, Sar4welch = sg.welch(xar4)

ax[1, 0].semilogy(pws[1:], Sar4[1:], label='periodogram')
ax[1, 0].semilogy(wws, Sar4welch, lw=2, label='Welch estimator')

ax[1, 1].semilogy(pws[1:], Sar4mod[1:], label='modified periodogram')
ax[1, 1].semilogy(wws,  Sar4welch, lw=2, label='Welch estimator')


ax[0, 0].set_title('AR2')
ax[0, 1].set_title('AR2')
ax[1, 0].set_title('AR4')
ax[1, 1].set_title('AR4')

ax[1, 0].set_xlabel(r'$\omega$')
ax[1, 1].set_xlabel(r'$\omega$')
ax[0, 0].set_ylabel(r'$\ln S(\omega)$')
ax[1, 0].set_ylabel(r'$\ln S(\omega)$')

xticks = [x/4 for x in range(0, 3)]
xlabels = [f'${x:1.2f}\\pi$' for x in xticks]

for _, a in np.ndenumerate(ax):
    a.legend()
    a.set_ylim((1e-6, 1e6))
    a.set_xticks(xticks, labels=xlabels)


fig.savefig('figs/w6-42_periodograms.pdf')
plt.show()
