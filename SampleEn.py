from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import EntropyHub as EH
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
def FuzzyEn2(s, r=0.2, m=2, n=2):
	th = r * np.std(s)
	return EH.FuzzEn(s, 2, r=(th, n))[0][-1]
def SampleEntropy2(Datalist, r=0.2, m=2):
	th = r * np.std(Datalist) #容限阈值
	return EH.SampEn(Datalist,m,r=th)[0][-1]
warnings.filterwarnings("ignore")
box=np.load("x_test.npy")
fs = 1000
#采样点数
num_fft = 28479;
list=[]
box2=pd.read_csv("spe2.csv",index_col=0)
# print(box2.values[0])
box2=box2.values
er=box2[:,0]
df_loess_15 = pd.DataFrame(lowess(box2.value, np.arange(len(box2.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])
"""
生成原始信号序列

在原始信号中加上噪声
np.random.randn(t.size)
"""
t = np.arange(0, 1, 1/fs)
f0 = 100
f1 = 200
for index in range(38):


    x = box[index]
    print(len(x))
    Y = fft(x, num_fft)
    Y = np.abs(Y)
    list.append(SampleEntropy2(Y))
    print(SampleEntropy2(Y))
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(511)
    ax.set_title('original signal')
    plt.tight_layout()
    plt.plot(x)
    #
    # """
    # FFT(Fast Fourier Transformation)快速傅里叶变换
    # """
    # Y = fft(x, num_fft)
    # Y = np.abs(Y)
    #
    # ax = plt.subplot(512)
    # ax.set_title('fft transform')
    # plt.plot(20 * np.log10(Y[:num_fft // 2]))
    #
    # """
    # 功率谱 power spectrum
    # 直接平方
    # """
    # ps = Y ** 2 / num_fft
    # ax = plt.subplot(513)
    # ax.set_title('direct method')
    # plt.plot(20 * np.log10(ps[:num_fft // 2]))
    #
    # """
    # 相关功谱率 power spectrum using correlate
    # 间接法
    # """
    # cor_x = np.correlate(x, x, 'same')
    # cor_X = fft(cor_x, num_fft)
    # ps_cor = np.abs(cor_X)
    # ps_cor = ps_cor / np.max(ps_cor)
    # ax = plt.subplot(514)
    # ax.set_title('indirect method')
    # plt.plot(20 * np.log10(ps_cor[:num_fft // 2]))
    # plt.tight_layout()
    plt.show()

print(list)