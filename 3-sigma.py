import numpy as np
import antropy as ant
from pyts.approximation import PiecewiseAggregateApproximation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import numpy as np
from sklearn.decomposition import PCA
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
np.random.seed(1234567)
x = np.zeros((1,3000))
x=np.squeeze(x)
list1=[]
boxes3 = read_pickle("machine-1-1_test.pkl")
boxes3=np.squeeze(np.array(boxes3))
for j in range(len(boxes3[0,:])):
    qlist = []
    te = boxes3[:, j]  # 保存基本统计量
    maximum = max(te)
    minimum = min(te)
    # median = statistics.median(te)
    # QL = np.quantile(te, 0.25, interpolation='lower')  # 下四分位数
    # QU = np.quantile(te, 0.75, interpolation='higher')  # 上四分位数
    # R=maximum-minimum
    # IQR = QU - QL
    # threshold1 = QL - 1.5 * (IQR) # 下阈值
    # threshold2 = QU + 1.5 * (IQR)  # 上阈值
    ystd = np.std(te)
    ymean = np.mean(te)
    threshold1 = ymean - 3 * ystd
    threshold2 = ymean + 3 * ystd
    # print(1.5 * (IQR+R/4))
    # print(threshold1)
    # print(threshold2)
    # print(IQR)
    # print(median)
    flag = 0
    hs = 0
    templ=[]
    x = np.linspace(1, 28479, num=28479)
    plt.tick_params(labelsize=12)
    plt.plot(x, te)
    plt.axhline(y=threshold2, c="r", ls="--", lw=2)
    outlier = []  # 将异常值保存
    outlier_x = []

    for i in range(0, len(te)):
        if (te[i] < threshold1) | (te[i] > threshold2):
            outlier.append(te[i])
            # outlier_x.append(data_x[i])
            outlier_x.append(x[i])
    plt.plot(outlier_x, outlier, 'yo')
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('KPI Value', fontsize=16)
    plt.show()
    for i in range(0, len(te)):
        if (te[i] < threshold1) | (te[i] > threshold2):
            # print(abs(te[i] - median) )
            # print(te[i])
            flag = flag + 1
            templ.append(i)
            qlist.append(1)

            # if (R != 0):
            #
            #     hs = hs + abs(te[i] - median) / (R/4)
        else:
            qlist.append(0)
                # print(abs(te[i] - median))
    plt.tick_params(labelsize=12)
    plt.plot(x, qlist)
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('Label Value', fontsize=16)
    plt.show()
    temp=ant.sample_entropy(qlist)
    list1.append(temp)



# # Permutation entropy
# print(ant.perm_entropy(x, normalize=True))
# # Spectral entropy
# print(ant.spectral_entropy(x, sf=100, method='welch', normalize=True))
# # Singular value decomposition entropy
# print(ant.svd_entropy(x, normalize=True))
# # Approximate entropy
# print(ant.app_entropy(x))
# # Sample entropy
# # Hjorth mobility and complexity
# print(ant.hjorth_params(x))
# # Number of zero-crossings
# print(ant.num_zerocross(x))
# # Lempel-Ziv complexity
# print(ant.lziv_complexity('01111000011001', normalize=True))