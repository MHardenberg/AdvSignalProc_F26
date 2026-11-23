#!/usr/bin/env python

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt


def signal_gen(nchunks: int, fs: float, fcarrier: float,
               frep: float, time_const: float, duty_cycle: float):
    chunck_time = 1./frep
    chunk_samples = int(chunck_time * fs)
    dead_samples = int(chunk_samples*(1 - duty_cycle))

    chunk = np.zeros(chunk_samples)
    ns_chunk = np.arange(chunk_samples)

    # mask
    mask = np.ones_like(chunk)
    mask[:dead_samples] = 0
    mask = np.tile(mask, nchunks)

    # clean signal
    chunk[dead_samples:] = \
        np.exp(-time_const * ns_chunk[:chunk_samples - dead_samples])\
        * np.cos(np.pi*fcarrier / fs * ns_chunk[:chunk_samples-dead_samples])

    return np.tile(chunk, nchunks)


def mask_gen(nchunks: int, fs: float, frep: float, duty_cycle: float):
    chunck_time = 1./frep
    chunk_samples = int(chunck_time * fs)
    dead_samples = int(chunk_samples*(1 - duty_cycle))

    chunk = np.zeros(chunk_samples)

    # mask
    mask = np.ones_like(chunk)
    mask[:dead_samples] = 0
    return np.tile(mask, nchunks)


def gen_bit_stream(npoints: int, fs: float, fenvelope_noise: float):
    number_of_bits = int(npoints / fs * fenvelope_noise)
    return np.random.randint(2, size=number_of_bits)


def gen_am_signal(npoints: int, fs: float, fcarrier: float,
                  fenvelope: float, bits: np.ndarray,
                  timing_offset: float, phase_offset: float):
    signal = np.cos(2*np.pi*fcarrier/fs*np.arange(npoints)+phase_offset)
    modulated_signal = np.zeros_like(signal)

    bit_width = int(fs/fenvelope)
    bit_window = sg.windows.hann(bit_width)

    for i, b in enumerate(bits):
        if b == 0:
            continue
        start_idx = min(i*bit_width, npoints-1)
        end_idx = min((i+1)*bit_width, npoints)

        modulated_signal[start_idx:end_idx] = \
            signal[start_idx:end_idx] * bit_window

    step_offset = timing_offset * fs
    modulated_signal = np.roll(modulated_signal, step_offset)
    return modulated_signal


def measure():
    # this is the signal we want
    dn = signal_gen(nchunks, fs, fcarrier_signal, frep,
                    time_const, duty_cycle)

    # these are the 'holes' introduced by our measurement cycle
    mask = mask_gen(nchunks, fs, frep, duty_cycle)

    # the logical bit stream
    bits = gen_bit_stream(len(dn), fs, fenvelope_noise)

    # am signal with randomised phase, timing, and amplitude
    A = (np.random.rand() + 1)/10  # between .1 - .2
    dt = (np.random.rand()-.5) / fenvelope_noise  # plus minus half a period
    phi = np.random.rand() * 2*np.pi  # fully random phase
    am_signal = A*gen_am_signal(
        len(dn), fs, fcarrier_noise, fenvelope_noise, bits, timing_offset=dt,
        phase_offset=phi)

    xn = mask*(dn + 0.1*am_signal +
               np.random.normal(loc=0, scale=.1, size=len(dn)))

    # measured signal
    return bits, mask, xn, dn


if __name__ == "__main__":
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
    bits, mask, xn, dn = measure()

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
    ...
