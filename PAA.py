from pyts.approximation import PiecewiseAggregateApproximation
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import read_pickle
def PAA(S, SegSize):

   L = len(S)  # the length of the time series
   C_S = S  # a copy of S
   Terms = math.ceil(L / SegSize)
   segment = list(range(0, L, SegSize))

   Mean_Segment_S = np.zeros(Terms, float)
   Max_Segment_S = np.zeros(Terms, float)

   for i in range(0, Terms - 1):
      Mean_Segment_S[i] = np.mean(C_S[segment[i]: segment[i + 1]])
      Max_Segment_S[i] = np.max(C_S[segment[i]: segment[i + 1]])
   i = Terms - 1
   Mean_Segment_S[i] = np.mean(C_S[segment[i]: L])
   DS = Mean_Segment_S
   return DS

boxes3 = read_pickle("data_test.pkl")
boxes3=np.squeeze(np.array(boxes3),axis=0)
list=[]
for i in range(0,38):
   X = boxes3[:, i]
   x1 = np.linspace(1, 28479, num=28479)
   x2 = np.linspace(1, 475, num=475)

   n_timestamps = len(X)
   window_size = 60
   transformer = PiecewiseAggregateApproximation(window_size=window_size)
   X_transform = transformer.transform(X.reshape(1, -1))  # 转换
   list.append(X_transform[0])
   plt.plot(x1,X,label='Original')
   plt.plot(np.arange(window_size // 2,n_timestamps + window_size // 2,window_size),X_transform[0],'o--',label='PAA')
   plt.legend(loc='best', fontsize=10)
   plt.xlabel('Time (min)', fontsize=16)
   plt.ylabel('KPI Value', fontsize=16)
   plt.savefig('savefig_example.png')

   plt.show()

np.save('test0.npy',list)