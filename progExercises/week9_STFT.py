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

    Nsft = SFT.p_num(N)
    stft = SFT.stft(xn)
    t_lo, t_hi = SFT.extent(N)[:2]  # time range
    f_lo, f_hi = SFT.extent(N)[2:]  # f range

    freqs = np.empty_like(stft, dtype=float)

    for i in range(freqs.shape[1]):
        freqs[:, i] = np.linspace(f_lo, f_hi, stft.shape[0])

    ts = np.arange(Nsft) * SFT.delta_t / fs
    meanfreqs = np.average(freqs, weights=np.abs(
        stft) / np.max(np.abs(stft)), axis=0)

    extend = list(SFT.extent(N))

    img = ax.imshow(np.abs(stft), origin='lower',
                    aspect='auto', extent=extend, cmap='Reds')
    fig.colorbar(img, label="Magnitude $|S_x(t, f)|$")

    ax.plot(ts, meanfreqs, 'k',  label=r"$f_\mu$")
    ax.legend()

    return fig


def yn(ns: np.ndarray, alpha: float):
    yn = np.zeros_like(ns)
    seg_a = np.multiply(0 <= ns, ns < 20e3)
    seg_b = np.multiply(20e3 <= ns, ns < 25e3)
    seg_c = 25e3 <= ns

    yn[seg_a] = np.cos((alpha*ns[seg_a])**2)
    yn[seg_b] = np.cos(.2*np.pi*ns[seg_b])
    yn[seg_c] = np.cos(.2*np.pi*ns[seg_c]) + np.cos(.23*np.pi*ns[seg_c])

    return yn


def main():

    ns = np.arange(int(30e3), dtype=float)
    xn = yn(ns, 0.005)

    fig = dtft(xn, fs=1)
    fig.savefig('figs/w9-stft.chirpstftpdf')
    plt.show()


if __name__ == "__main__":
    main()
