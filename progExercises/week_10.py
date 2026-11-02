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



def proces(xn: np.ndarray) -> np.ndarray:
    yn = np.zeros_like(xn)

    for i in range(len(xn)):
        yn[i] = xn[i]
        if i >= 1:
            yn[i] -= 0.25 * yn[i - 1]
        if i >= 2:
            yn[i] += 0.5 * yn[i - 2]

    return yn


def LMSupdate(
    wcoeff: np.ndarray,
    xk: np.ndarray,
    err: float,
    step: float,
) -> np.ndarray:
    return wcoeff + 2 * step * err * xk


def predict(wcoeff: np.ndarray, xk: np.ndarray) -> float:
    return float(wcoeff @ xk)


def main():
    step = 1e-3
    filter_order = 10

    vn = np.random.normal(size=1000)
    xn = proces(vn)

    yn = np.zeros_like(xn)
    err = np.zeros_like(xn)
    w = np.zeros(filter_order)

    for i in range(filter_order - 1, len(xn)):
        xk = vn[i - filter_order + 1 : i + 1][::-1]
        yn[i] = predict(w, xk)
        err[i] = xn[i] - yn[i]
        w = LMSupdate(w, xk, err[i], step)

    return yn, xn, err, w


if __name__ == "__main__":
    yn,xn, err, w = main()
    fig, ax = plt.subplots(nrows = 2, figsize=(18, 8))
        

    ax[0].set_title("System identification")
    ax[0].set_ylabel("amplitude (arb.u.)")

    ax[0].plot(xn, label="x[n]", lw=2)
    ax[0].plot(yn, label="y[n]", lw=2)


    ax[1].plot((err/ np.max(xn))**2, label="SE", lw=2)
    ax[1].set_ylabel("error amplitude (arb.u.)")
    ax[1].set_xlabel("sample")


    for a in ax:
        a.legend()
    fig.savefig("figs/w10_lms.pdf")
    plt.show()

