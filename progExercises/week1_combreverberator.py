#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.io import wavfile
import pygame


plt.switch_backend('qtagg')
plt.style.use('classic')

# Set LaTeX rendering and font settings
rc('text', usetex=True)
rc('font', family='serif', serif='Computer Modern', size=20)

''' Helpers ----------------------------------------------------------------'''


def plot_lollipop(ax,  y: np.ndarray, x: np.ndarray | None = None,
                  label: str | None = None, color: str = 'k', alpha: float = 1):
    if x is None:
        x = np.arange(len(y))
    ax.plot([x[0], x[-1]], [0, 0], 'k')
    ax.scatter(x, y, color=color, fc='w', lw=2, s=50, label=label, alpha=alpha)
    ax.vlines(x, np.zeros_like(y), ymax=y, color=color, lw=2, alpha=alpha)


def play_array(x: np.ndarray, fs: int = 48000):
    pygame.mixer.pre_init(fs, size=-16, channels=1)
    pygame.mixer.init()

    sound_array = x.copy()

    # pygame epects int16 - so we scale the thing to +- 1 and then make it fit
    # the int16 range
    sound_array /= np.max(np.abs(sound_array))
    sound_array = np.int16(sound_array * (np.iinfo(np.int16).max-1))
    sound = pygame.sndarray.make_sound(sound_array)

    sound_player = sound.play()
    while sound_player is not None and sound_player.get_busy():
        pygame.time.Clock().tick(60)


def reverb_io(xn: np.ndarray, D: int, a: float) -> np.ndarray:
    assert (-1 < a < 1)  # stability
    yn = np.zeros_like(xn)
    for n, x in enumerate(xn):
        yn[n] = x + (a*yn[n - D] if D <= n else 0)
    return yn


''' Exercises --------------------------------------------------------------'''


def reverb_exercise():
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    D = 5
    a = .7
    N = 100

    ''' Impulse rsp '''
    xn = np.zeros(N)
    xn[0] = 1
    yn = reverb_io(xn, D, a)

    # label dummies
    ax1.scatter([], [],  color='k', label='x[n]')
    ax1.scatter([], [],  color='r', label='y[n]')

    plot_lollipop(ax1,  xn, color='k')
    plot_lollipop(ax2,  yn, color='r')

    ax1.legend()
    ax1.set_xlabel('n')
    ax1.set_ylabel('h[n]')

    ax2.set_xlabel('n')
    ax2.set_ylabel('h[n]')

    fig.savefig('figs/w1_reverb.pdf')
    plt.show()

    ''' Magnitude rsp '''
    fig, ax = plt.subplots()
    yn_fft = np.fft.fft(yn)

    ax.plot(np.linspace(-np.pi, np.pi, N), np.abs(yn_fft), 'k', lw=2)

    xticks = [np.pi * x/2 for x in range(-2, 3)]
    xlabels = [f'${x/np.pi:1.2f}\\pi$' for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_title('Magnitude responce')
    ax.set_ylabel(r'$|H\left(e^{j\omega}\right)|$')
    ax.set_xlabel(r'$\omega$')
    fig.savefig('figs/w1_reverbMag.pdf')
    plt.show()


def filtered_wav_exercise():
    fs, xs = wavfile.read('./data/Music_sample_1(48kHz).wav')
    xs_chnl1 = xs[:, 0].astype(np.float32) / np.iinfo(np.int16).max
    xs_reverb = reverb_io(xs_chnl1, D=80000, a=0.7)

    # unfiltered
    play_array(xs_chnl1, fs=fs)

    # horrible
    play_array(xs_reverb, fs=fs)


if __name__ == '__main__':

    ''' Uncomment what you want!'''
    # filtered_wav_exercise()

    # reverb_exercise()

    pass
