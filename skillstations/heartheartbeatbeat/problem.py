#!/usr/bin/env python

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt


def main():
    data = np.genfromtxt('./heartheartbeatbeat/eeccgg.csv',
                         delimiter=',', dtype=float, skip_header=1)

    tn = data[:, 0]
    xn = data[:, 1]

    _, ax = plt.subplots()
    ax.plot(tn, xn, label="measured signal", lw=2)
    ax.set_ylabel("amplitude (arb.u)")
    ax.set_xlabel("time (s)")
    ax.set_title(
        "Heartbeat but the doctor also got real nervous" +
        " and you can see his aswell (and some more nonsense)")
    ax.legend()
    plt.show()

    """ ----------------------------------------------------------------------
        You have the tn and xn samples of a double heartbeat.
        There is some heavy powerline pollution and the modulation of
        a very mild earthquake that just happened to happen at the same time.

        Try to clean up the double heart rythm
    """

    """ Your code ( or import csv into matlab) """
    """ ------------------------------------------------------------------- """

    ...


if __name__ == "__main__":
    main()
