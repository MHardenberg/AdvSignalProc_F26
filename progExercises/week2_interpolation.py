#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


plt.switch_backend('qtagg')
plt.style.use('classic')

# Set LaTeX rendering and font settings
rc('text', usetex=True)
rc('font', family='serif', serif='Computer Modern', size=20)

''' Helpers ----------------------------------------------------------------'''


def plot_lollipop(ax,  y: np.ndarray, x: np.ndarray | None = None,
                  label: str | None = None, color: str = 'k',
                  alpha: float = 1):
    if x is None:
        x = np.arange(len(y))
    if len(y) == 0:
        ax.scatter([], [], color=color, fc='w', lw=2,
                   s=50, label=label, alpha=alpha)
        return

    ax.plot([x[0], x[-1]], [0, 0], 'k')
    ax.scatter(x, y, color=color, fc='w', lw=2, s=80, label=label, alpha=alpha)
    ax.vlines(x, np.zeros_like(y), ymax=y, color=color, lw=3, alpha=alpha)


''' Signal ----------------------------------------------------------------'''


def signal(t: np.ndarray | int) -> np.ndarray | int:
    return np.sin(.5*(t - 3)) * np.exp(-(t-3)/10) * np.heaviside(t-3, 0)


def upsampled(xn: np.ndarray, I) -> np.ndarray:
    xU = np.zeros(len(xn) * I)
    xU[::I] = xn
    return xU


def ideal_interpol(xn: np.ndarray, I: int) -> np.ndarray:
    xI = upsampled(xn, I)
    for m in range(len(xI)):
        # numpy's sinc defined as sinc(x) = sin(pi *x) / pi*x
        xI[m] = np.sum([x * np.sinc((m - n*I)/I)
                        for n, x in enumerate(xn)])
    return xI


def mean_square_error(ns: np.ndarray, xn: np.ndarray, T: float) -> float:
    return np.mean((signal(ns*T) - xn)**2)


def main():
    I = 2
    ts = np.linspace(0, 50, 100)
    ns = np.arange(50)
    fig, ax = plt.subplots()

    T = 1
    xc = signal(ts)
    xn = signal(ns * T)

    ''' Upsampling '''
    xU = upsampled(xn, I)
    nU = np.arange(len(xU))

    ''' Interpolation '''
    xI = ideal_interpol(xn, I)
    xI_err = mean_square_error(nU, xI, T)

    ''' Plotting '''
    ax.plot(ts*I, xc, "k", label=r"$x_C(t)$")
    plot_lollipop(ax, xI, nU, color="r",
                  label=f"$x_I[n] (MSE = {round(xI_err, 2)})$")
    plot_lollipop(ax, xn, ns * I, color="k", label=r"$x_U[n]$")

    ax.set_xlim((I*ts[0], I*ts[-1]))
    ax.set_ylim((1.2 * np.min(xc), 1.2 * np.max(xc)))
    ax.legend(loc="lower right")

    plt.savefig('figs/w2_interpolator.pdf')
    plt.show()


if __name__ == "__main__":
    main()
