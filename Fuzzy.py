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
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
def read_pickle(work_path):
   data_list = []
   with open(work_path, "rb") as f:
      while True:
         try:
            data = pickle.load(f)
            data_list.append(data)
         except EOFError:
            break
   return data_list
def FuzzyEn2(s, r=0.2, m=2, n=2):
	th = r * np.std(s)
	return EH.FuzzEn(s, 2, r=(th, n))[0][-1]
def SampleEntropy2(Datalist, r=0.2, m=2):
	th = r * np.std(Datalist) #容限阈值
	return EH.SampEn(Datalist,m,r=th)[0][-1]
warnings.filterwarnings("ignore")
boxes3 = read_pickle("msl_test.pkl")
boxes3=np.squeeze(np.array(boxes3),axis=0)
box=np.load("x_test.npy")
fs = 1000
#采样点数
num_fft = 28479;
list=[]
box2=pd.read_csv("spe2.csv",index_col=0)
# print(box2.values[0])
box=np.load("x_test.npy")
box2=box2.values
er=box2[:,17]
er=box[10]
# x=np.arange(len(er))
# df_loess_15 = lowess(er, np.arange(len(er)), frac=0.02)[:, 1]
# print(df_loess_15)
# plt.clf()
# plt.plot(x, er, label='y noisy')
# plt.plot(x, df_loess_15 , label='y pred')
# plt.legend()
# plt.show()
result_mul = seasonal_decompose(boxes3[:,33],  model='additive', extrapolate_trend='freq', period=47)
x=np.arange(len(boxes3[:,33]))
seasonal = result_mul.seasonal
qlist=boxes3[:,33]-seasonal
# plt.plot(x, boxes3[:,11]-seasonal , label='y pred')
# plt.show()
print(SampleEntropy2(qlist))
# print(result_mul)
# result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
# plt.show()


# x=np.arange(len(boxes3[:,12]))
# df_loess_15 = lowess(boxes3[:,12], x, frac=0.005)[:, 1]
#
# plt.clf()
# plt.plot(x, boxes3[:,12], label='y noisy')
# plt.plot(x, df_loess_15 , label='y pred')
# plt.legend()
