#!/usr/bin/env python

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

import measurement as m


def main():
    nchunks = 25
    fs = 48e3
    frep = 10
    time_const = 5e-3
    duty_cycle = .9

    fcarrier_signal = 23.5e3
    fcarrier_noise = 21.8e3
    fenvelope_noise = 1e2

    np.random.seed(0)

    """ Measurement """
    bits, mask, xn, dn = m.measure(
        nchunks,
        fs,
        frep,
        time_const,
        duty_cycle,
        fcarrier_signal,
        fcarrier_noise,
        fenvelope_noise,
    )

    _, ax = plt.subplots()
    ax.plot(np.arange(len(xn))/fs, xn, label="measured signal", lw=2)
    ax.set_ylabel("amplitude (arb.u)")
    ax.set_xlabel("sample")
    ax.set_title("AM noise in gappy signal")
    ax.legend()
    plt.show()

    """
    You have measured:
        x[n] = observed_signal (desired signal + am signal + white noise)
        m[n] = mask based on experiment (you know where the holes are)
        bits = decoded bit stream

    Known constants:
        nchunks = 25
        fs = 48e3
        frep = 10
        time_const = 5e-3
        duty_cycle = .9

        fcarrier_signal = 23.5e3
        fcarrier_noise = 21.8e3
        fenvelope_noise = 1e2

    You want to eliminate the am signal
    without interfering with the desired signal

    Youre free to try any technique we've seen
    (maybe you can adapt previously written code)

    You don't know dn. You can only use it to score against a ground truth
    """

    """ Your code ( or import csv into matlab) """
    """ ------------------------------------------------------------------- """

    ...


if __name__ == "__main__":
    main()
