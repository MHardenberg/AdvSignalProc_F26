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


N = 100
xna = np.zeros(N)
xnb = np.zeros(N)
n = np.arange(N)

for xn in (xna, xnb):
    xn[0] = 1 if np.random.random() >= 1 else -1

    for i, x in enumerate(xn[1:]):
        # change sign with 75 % probability
        if x == 1:
            xn[i + 1] = -1 if np.random.random() < .75 else 1
        # change sign with 25 % probability
        else:
            xn[i + 1] = 1 if np.random.random() < .25 else -1

fig, ax = plt.subplots(2)
ax[0].stairs(xna,  lw=2, color='k')
ax[1].stairs(xnb,  lw=2, color='r')

ax[0].set_title('Random process realisation')
ax[0].set_xlabel(r'$n$')
ax[0].set_ylabel(r'$x[n]$')
ax[1].set_ylabel(r'$x[n]$')

ax[0].set_ylim((0, 1.25))
ax[1].set_ylim((0, 1.25))
fig.savefig('figs/w5_rndprocess.pdf')
plt.show()
