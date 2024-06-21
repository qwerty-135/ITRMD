import numpy as np
import antropy as ant
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


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


start = time.time()
list1 = []
listz = [4, 7, 16, 17, 36, 37]  # variance = 0
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
    for i in range(0, len(te)):
        if (te[i] < threshold1) | (te[i] > threshold2):
            flag = flag + 1
            templ.append(i)
            qlist.append(1)

        else:
            qlist.append(0)

    temp = ant.sample_entropy(qlist)
    if (j not in listz):
        list1.append(temp)

end = time.time()

ystd = np.std(list1)
ymean = np.mean(list1)
threshold1 = ymean - 3 * ystd
threshold2 = ymean + 3 * ystd

QL = np.quantile(list1, 0.25, interpolation='lower')
QU = np.quantile(list1, 0.75, interpolation='higher')
R = maximum - minimum
IQR = QU - QL
threshold1 = QL - 1.5 * (IQR)
threshold2 = QU + 1.5 * (IQR)

x1 = np.linspace(0, 37, num=38)
x2 = np.linspace(1, 38, num=38)
x1 = list(set(x1).difference(set(listz)))
# Draw plot
fig, ax = plt.subplots(figsize=(16, 10), dpi=80)

plt.axhline(y=threshold2, c="g", ls="--", lw=2)
ax.vlines(x=x1, ymin=0, ymax=list1, color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=x1, y=list1, s=75, color='firebrick', alpha=0.7)

plt.tick_params(labelsize=16)
ax.set_ylabel('Sample Entropy', fontsize=22)
ax.set_xlabel('KPI Dimensions', fontsize=22)
ax.set_xticks(x1)
plt.xticks(rotation=45)

plt.show()
