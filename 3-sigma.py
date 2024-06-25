
import matplotlib.pyplot as plt
import numpy as np
from utils import read_pickle

x = np.zeros((1, 3000))
x = np.squeeze(x)
list1 = []
boxes3 = read_pickle("data_test.pkl")
boxes3 = np.squeeze(np.array(boxes3))
for j in range(len(boxes3[0, :])):
    qlist = []
    te = boxes3[:, j]
    maximum = max(te)
    minimum = min(te)

    ystd = np.std(te)
    ymean = np.mean(te)
    threshold1 = ymean - 3 * ystd
    threshold2 = ymean + 3 * ystd

    flag = 0
    hs = 0
    templ = []
    x = np.linspace(1, 28479, num=28479)
    plt.tick_params(labelsize=12)
    plt.plot(x, te)
    plt.axhline(y=threshold2, c="r", ls="--", lw=2)
    outlier = []
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
            flag = flag + 1
            templ.append(i)
            qlist.append(1)

        else:
            qlist.append(0)
    plt.tick_params(labelsize=12)
    plt.plot(x, qlist)
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('Label Value', fontsize=16)
    plt.show()

