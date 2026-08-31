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


''' Helpers ----------------------------------------------------------------'''


def plot_lollipop(ax,  y: np.ndarray, x: np.ndarray | None = None,
                  label: str | None = None, color: str = 'k', alpha: float = 1):
    if x is None:
        x = np.arange(len(y))
    ax.plot([x[0], x[-1]], [0, 0], 'k')
    ax.scatter(x, y, color=color, fc='w', lw=2, s=50, label=label, alpha=alpha)
    ax.vlines(x, np.zeros_like(y), ymax=y, color=color, lw=2, alpha=alpha)


def main():
    num = 1., .23, .57
    denom = .57, .23, 1

    ts, rsp = sg.impulse((num, denom))
    imp = np.zeros_like(ts)
    imp[0] = 1.

    fig, ax = plt.subplots()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Amplitude (unitless)")
    plot_lollipop(ax, imp, ts, color="k", label="impulse")
    plot_lollipop(ax,  rsp, ts, color="r", label="impulse response")
    plt.legend()
    plt.savefig('figs/w1_reverbMag.pdf')
    plt.show()


if __name__ == "__main__":
    main()
