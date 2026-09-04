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


def cheby_lowpass(cutoff: float, fs: float, order: int = 8,
                  ripple_Db: float = .05):
    return sg.cheby1(order, ripple_Db, cutoff, 'low', fs=fs)


def cheby_lowpass_filter(xn: np.ndarray, cutoff: float,
                         fs: float, order: int = 8):
    b, a = cheby_lowpass(cutoff, fs, order=order)
    yn = sg.lfilter(b, a, xn)
    return yn


''' Exercise ---------------------------------------------------------------'''


def downsample(xn: np.ndarray, D: int, k: int = 0) -> np.ndarray:
    xD = xn[::D].copy()
    xD[:k] = 0
    return xD


def decimate(xn: np.ndarray, D: int) -> np.ndarray:
    cutoff = .8/D

    yn = cheby_lowpass_filter(xn, cutoff, 1)
    return downsample(yn, D, 1)


def prob12_21():
    N = 50
    D = 4

    n = np.arange(0, N)
    xn = np.cos(.2*np.pi * n)

    xd0 = downsample(xn, D, 0)
    xd1 = downsample(xn, D, 2)
    nD = np.arange(0, N, D)

    fig, axs = plt.subplots(nrows=3)

    plot_lollipop(axs[0], xn, n, color="k", label="$x[n]$")
    plot_lollipop(axs[1], xd0, nD, color="r", label=r"$x_{D=4, k=0}[n]$")
    plot_lollipop(axs[2], xd1, nD, color="b", label=r"$x_{D=4, k=2}[n]$")

    for ax in axs:
        ax.legend()
        ax.set_ylabel("x[n]")
    axs[-1].set_xlabel("n")
    plt.savefig('figs/w2_downsampling.pdf')
    plt.show()


def prob12_22():
    N = 100
    D = 2

    n = np.arange(0, N)
    xn = np.cos(.2*np.pi * n)

    xD = decimate(xn, D)
    nD = np.arange(0, N, D)

    fig, axs = plt.subplots(nrows=2)

    plot_lollipop(axs[0], xn, n, color="k", label="$x[n]$")
    plot_lollipop(axs[1], xD, nD, color="r", label=r"$x_{D=4, k=0}[n]$")

    for ax in axs:
        ax.legend()
        ax.set_ylabel("x[n]")
    axs[-1].set_xlabel("n")
    plt.savefig('figs/w2_decimation.pdf')
    plt.show()


def main():
    prob12_21()
    prob12_22()


if __name__ == "__main__":
    main()
