#!/usr/bin/env python

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from matplotlib import rc

plt.switch_backend('qtagg')
plt.style.use('classic')

# Set LaTeX rendering and font settings
rc('text', usetex=True)
rc('font', family='serif', serif='Computer Modern', size=16)


def component(n, t):
    return 2 / (n*np.pi) * (-1)**(1+n) * np.cos(n * t)


t = np.linspace(-np.pi, np.pi, 100)

x1 = sum([component(n+1, t) for n in range(1)])+.5
x2 = sum([component(n+1, t) for n in range(2)])+.5
x5 = sum([component(n + 1, t) for n in range(5)])+.5
x10 = sum([component(n + 1, t) for n in range(10)])+.5
x100 = sum([component(n+1, t) for n in range(200)]) + .5


_, ax = plt.subplots()


for x in [x1, x2, x5, x10, x100]:
    ax.plot(t, x)

plt.show()
