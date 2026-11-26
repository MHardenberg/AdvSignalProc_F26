#!/usr/bin/env python

import numpy as np


""" This files generate the weird a$$ data set - I'm sorry """


def gendata():
    np.random.seed(0)
    data = np.genfromtxt('./heartheartbeatbeat/ecg.txt',
                         delimiter=',', dtype=float, skip_header=1)
    tn = data[:, 0]
    dn = data[:, 1]
    # echo
    vn = np.zeros_like(dn)
    vn[100:] = np.roll(dn[100:], 100)
    vn = np.delete(vn, np.arange(0, len(vn),  21))

    xn = np.zeros_like(dn)
    xn += np.random.normal(loc=0, scale=.1, size=len(xn))
    xn *= np.cos(.5*np.pi*tn)
    xn += dn
    xn[:len(vn)] += vn
    xn += .5*np.cos(2*np.pi*50*tn)

    xn = xn[tn > 5]
    xn = xn[tn[:len(xn)] < 50]
    tn = tn[:len(xn)]

    csv = ["t(s), x(arb.u.)\n",]
    for i in range(len(tn)):
        csv += [f"{tn[i]}, {xn[i]}\n",]

    with open("./heartheartbeatbeat/eeccgg.csv", "w") as f:
        f.writelines(csv)


if __name__ == "__main__":
    gendata()
