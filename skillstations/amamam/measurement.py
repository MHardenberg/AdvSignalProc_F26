import numpy as np
import scipy.signal as sg


def _gen_bit_stream(npoints: int, fs: float, baud: float):
    number_of_bits = int(npoints / fs * baud)
    return np.random.randint(2, size=number_of_bits)


def _gen_am_signal(npoints: int, fs: float, fcarrier: float, baud: int,
                   bits: np.ndarray, timing_offset: float,
                   phase_offset: float):
    signal = np.cos(2*np.pi*fcarrier/fs*np.arange(npoints)+phase_offset)
    modulated_signal = np.zeros_like(signal)

    bit_width = int(fs/baud)
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


def measure(duration: float, fs: float):
    np.random.seed(0)
    xn = np.zeros(int(duration * fs), dtype=float)

    stations = {}
    stations["names"] = ["Larry AM", "Consider Phlebas",
                         "Is my paint dry yet?",]
    stations["bits"] = []
    stations["carriers"] = []
    stations["baud"] = [200, 100, 200]

    for i in range(3):
        baud = stations["baud"][i]
        stations["bits"] += [_gen_bit_stream(len(xn), fs, baud),]
        stations["carriers"] += [round(np.random.random(), 1) * 20e3 + 4e3,]

        timing_offset = (np.random.random() - .5)/baud  # pm half a period
        phase_offset = 2*np.pi*np.random.random()  # fully random phase
        amplitude = .5*np.random.random() + .5
        xn += amplitude*_gen_am_signal(len(xn), fs, stations["carriers"][i],
                                       baud, stations["bits"][i],
                                       timing_offset, phase_offset)

    csv = ["t(s), x(arb.u.)\n",]
    for i in range(len(xn)):
        csv += [f"{i / fs}, {xn[i]}\n",]

    with open("./amamam/amamam.csv", "w") as f:
        f.writelines(csv)
    return xn, stations
