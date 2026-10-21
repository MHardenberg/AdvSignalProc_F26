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


def dtft(xn: np.ndarray, fs: float = 1.):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlabel("time $t$ (s)")
    ax.set_ylabel("frequency $f$ (Hz)")

    g_std = 10
    N = len(xn)
    w = sg.windows.gaussian(101, std=g_std, sym=True)
    SFT = sg.ShortTimeFFT(w, hop=25, fs=fs,
                          mfft=401, scale_to='magnitude')

    stft = SFT.stft(xn)
    t_lo, t_hi = SFT.extent(N)[:2]  # time range
    f_lo, f_hi = SFT.extent(N)[2:]  # f range

    freqs = np.empty_like(stft, dtype=float)

    for i in range(freqs.shape[1]):
        freqs[:, i] = np.linspace(f_lo, f_hi, stft.shape[0])

    extend = list(SFT.extent(N))
    img = ax.imshow(np.abs(stft) / np.max(np.abs(stft)), origin='lower',
                    aspect='auto', extent=extend, cmap='Reds')
    fig.colorbar(img, label="Magnitude $|S_x(t, f)|$")

    return fig


def yn(xn: np.ndarray, alpha: float, k: int):
    yn = xn.copy()
    yn[k:] += alpha * xn[:-k]
    return yn


def main():
    N = int(10e3)
    ns = np.arange(N)
    xs = np.zeros(N)

    xs[ns % 500 < 200] = np.cos(ns[ns % 500 < 200])

    alpha = .33
    D = 256
    ys = yn(xs, alpha, D)

    figpsd, ax = plt.subplots(figsize=(16, 6))

    ws, fsxs = sg.welch(xs, nperseg=256)
    _, fsys = sg.welch(ys, nperseg=256)

    ax.semilogy(ws, fsxs, lw=2, label="x[n]")
    ax.semilogy(ws, fsys, lw=2,
                label=f"y[n] = x[n] + {round(alpha, 2)} * x[n - {D}]")

    ax.set_ylabel("Amplitude (arb.u.)")
    ax.set_xlabel(r"frequency ($\pi$)")
    ax.legend()
    figpsd.savefig('figs/w9-echo_psd.pdf')

    fig = dtft(xs, fs=1)
    figecho = dtft(ys, fs=1)

    fig.savefig('figs/w9-echo_in.pdf')
    figecho.savefig('figs/w9-echo_out.pdf')
    plt.show()


if __name__ == "__main__":
    main()
