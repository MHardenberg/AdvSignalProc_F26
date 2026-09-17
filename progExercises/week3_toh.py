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


def upsampled(xn: np.ndarray, I) -> np.ndarray:
    xU = np.zeros(len(xn) * I)
    xU[::I] = xn
    return xU


def ideal_interpol_impulse_response(ns: np.ndarray, D: int) -> np.ndarray:
    ideal = np.zeros(len(ns))
    ideal = np.sinc(np.pi*ns/D)
    return ideal


def gtoh_impulse_response(ns: np.ndarray, a: float, I: int) -> np.ndarray:
    gtoh = np.zeros(len(ns))
    first_condition = (np.abs(ns) <= I)
    second_condition = (np.abs(ns) > I) * (np.abs(ns) <= 2*I)

    gtoh[first_condition] = (a+2)*np.abs(ns[first_condition]/I)**3 - \
        (a+3)*np.abs(ns[first_condition]/I)**2 + 1

    gtoh[second_condition] = a*np.abs(ns[second_condition]/I)**3 - \
        5*a*np.abs(ns[second_condition]/I)**2 + \
        8*a*np.abs(ns[second_condition]/I) - 4*a

    return gtoh


def main():
    I = 3
    ns = np.arange(-100, 100)
    N = len(ns)

    ''' Impulse response '''
    ''' ------------------------------------------------------------------- '''
    h_gtoh = gtoh_impulse_response(ns, a=.5, I=I)
    h_ideal = ideal_interpol_impulse_response(ns, D=I)

    fig, ax = plt.subplots()
    plot_lollipop(ax, h_gtoh, ns, label=r'$h(g_{toh})[n]$')
    plot_lollipop(ax, h_ideal, ns, color='r', label=r'$h(i)[d]$')
    ax.legend()
    ax.set_title('Impulse response')
    plt.savefig('figs/w3_42toh.pdf')

    ''' Magnitude response '''
    ''' ------------------------------------------------------------------- '''
    H_gtoh = np.fft.fft(h_gtoh)
    H_ideal = np.fft.fft(h_ideal)
    ws = np.fft.fftfreq(N)

    fig, ax = plt.subplots()
    plt.plot(ws[:N//2], np.abs(H_gtoh[:N//2]), 'k',
             label=r'$H(g_{gtoh})(e^{j\omega})$')
    plt.plot(ws[:N//2], np.abs(H_ideal[:N//2]),
             'r', label=r'$H(i)(e^{j\omega})$')
    ax.legend()
    ax.set_title('Magnitude response')
    plt.savefig('figs/w3_42toh_magrsp.pdf')

    ''' Interpolation '''
    ''' ------------------------------------------------------------------- '''

    ns = np.arange(-N//2, N//2)
    ms = np.arange(-N//(2*I), N//(2*I))
    ts = np.linspace(ms[0], ms[-1], 1000)
    condition = (ts <= 50) * (ts >= 0)
    xs = np.zeros_like(ts)
    xs[condition] = np.sin(.5*np.pi*ts[condition])

    condition = (ms <= 50) * (ms >= 0)
    xD = np.zeros(len(ms))
    xD[condition] = np.sin(.5*np.pi*ms[condition])
    xU = upsampled(xD, I)

    x_gtoh = np.convolve(xU, h_gtoh, mode='same')
    x_ideal = np.convolve(xU, h_ideal, mode='same')

    fig, ax = plt.subplots()
    plt.plot(I*ts, xs, 'k', lw=2, label=r'$x(ts)$')
    plot_lollipop(ax, xU[:len(ns)],  ns, color='k', label=r'$x_U[n]$')
    plot_lollipop(ax, x_gtoh[:len(ns)], ns,
                  color='r', label=r'$x(g_{toh})[n]$')
    plot_lollipop(ax, x_ideal[:len(ns)], ns, color='b', label=r'$x(i)[n]$')
    ax.legend()
    ax.set_title('Interpolated')
    plt.savefig('figs/w3_42interpol.pdf')

    plt.show()


if __name__ == "__main__":
    main()
