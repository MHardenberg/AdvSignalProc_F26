#!/usr/bin/env python


import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from PIL import Image  # (pacakge name is pillow - i.e. pip3 install pillow )


def downsample_1d(xn: np.ndarray, D: int) -> np.ndarray:
    assert (xn.ndim == 1)
    return xn[::D]


def decimate_1d(xn: np.ndarray, D: int) -> np.ndarray:
    cutoff = .8/D
    yn = sg.cheby_lowpass_filter(xn, cutoff, 1)
    return downsample_1d(yn, D, 1)


def upsample_1d(xn: np.ndarray, U: int) -> np.ndarray:
    assert (xn.ndim == 1)
    yn = np.zeros(len(xn) * U)
    yn[::U] = xn

    return yn


def interpolate_1d(xn: np.ndarray, I: int, type: str = "linear") -> np.ndarray:
    xI = upsample_1d(xn, I)
    match type:
        case "linear":
            ...

            return xI

        case "sinc":
            for m in range(len(xI)):
                # numpy's sinc defined as sinc(x) = sin(pi *x) / pi*x
                xI[m] = np.sum([x * np.sinc((m - n*I)/I)
                                for n, x in enumerate(xn)])
            return xI

        case _:
            raise (ValueError("invalid"))


def main():
    filename = './downup/roddy.bmp'

    img = Image.open(filename).convert('L')
    roddy = np.array(img)
    print('image shape', roddy.shape)

    plt.imshow(roddy, cmap='gray')
    plt.show()

    """ Up and downsampling of images """
    """ This is a simple example of resampling of images
        You need to test the out put of the up and downsampling methods row 
        by row.

        Investigate:
            Whats the pixel spectrom along the x-axis?
            How well does the linear vs the sinc interpolator?
            Can you see sampling artifacts, if you forget to lowpass filter
            before downsampling?

            What happens if you decimate in the fourier domain?

    """

    # Down sample and show
    ...

    # Upsample and show
    ...

    # Interpolate and show
    ...

    # Fourier domain analysis
    ...


if __name__ == "__main__":
    main()
