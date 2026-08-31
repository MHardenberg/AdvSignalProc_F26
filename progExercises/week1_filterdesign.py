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


def filter(frq: np.ndarray, num: tuple[float], denom: tuple[float]) -> np.ndarray:
    out_num = np.zeros(len(frq), dtype=complex)
    out_denom = np.zeros(len(frq), dtype=complex)

    order = len(num) - 1
    exp_frq = np.exp(1j * frq.astype(complex))

    for i, n in enumerate(num):
        out_num += n * exp_frq**(order - i)

    for i, d in enumerate(denom):
        out_denom += d * exp_frq**(order - i)

    return out_num / out_denom


def main():
    wall_stop = .4, .95

    frq = np.linspace(0, np.pi, 100)

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.set_ylabel("Amplitude (unitless)")
    ax1.set_ylabel("Group delay")
    ax1.set_xlabel(r"$\omega$ ($\pi$)")

    for order in (1, 2, 4, 8):
        num, denom = sg.iirfilter(order, wall_stop, btype='bandstop')
        frq_rsp = filter(frq, num, denom)
        _, grpdelay = sg.group_delay((num, denom), w=frq)
        ax0.plot(frq/np.pi, np.abs(frq_rsp),
                 label=f"O = {order}", lw=2)

        ax1.plot(frq/np.pi, grpdelay,
                 label=f"O = {order}", lw=2)
    # plot_lollipop(ax, imp, ts, color="k", label="impulse")
    # plot_lollipop(ax,  rsp, ts, color="r", label="impulse response")
    plt.legend()
    plt.savefig('figs/w1_filterDesign_freq.pdf')
    plt.show()


if __name__ == "__main__":
    main()
