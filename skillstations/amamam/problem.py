#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import measurement as m


def main():
    xn, stations = m.measure(60, 48e3)
    names = stations["names"]
    carriers = stations["carriers"]
    baud = stations["baud"]

    print("Recieved stations:")
    for n, c, b in zip(names, carriers, baud):
        print(f"{n}: {c*1e-3} kHz at {b} baud")
    _, ax = plt.subplots()
    ax.plot(xn)
    plt.show()

    """
        Distinquish the three AM radio stations and decode their bit stream
        You have: their carriers and baud rates

        sample rate = 48kHz
        duration = 60 s

        Use stations["bits"] as a ground truth to compare your result to
    """

    """ Your code ( or import csv into matlab) """
    """ ------------------------------------------------------------------- """

    ...


if __name__ == "__main__":
    main()
