from scipy.fftpack import fft, fftshift, ifft
import matplotlib.pyplot as plt
import warnings
import EntropyHub as EH
import pandas as pd
import numpy as np


def FuzzyEn(s, r=0.2, m=2, n=2):
    th = r * np.std(s)
    return EH.FuzzEn(s, 2, r=(th, n))[0][-1]



def SampleEntropy(Datalist, r=0.2, m=2):
    th = r * np.std(Datalist)
    return EH.SampEn(Datalist, m, r=th)[0][-1]


warnings.filterwarnings("ignore")
box = np.load("x_test.npy")
fs = 1000

num_fft = 28479;
list = []


t = np.arange(0, 1, 1 / fs)
f0 = 100
f1 = 200
for index in range(38):
    x = box[index]

    Y = fft(x, num_fft)
    Y = np.abs(Y)
    list.append(SampleEntropy(Y))

    ax = plt.subplot(511)
    ax.set_title('original signal')
    plt.tight_layout()
    plt.plot(x)

    plt.show()
print(list)
