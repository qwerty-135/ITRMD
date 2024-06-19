import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mean_shift(data, bandwidth=3):
    labels = -1 * np.ones(len(data))
    centroids = []
    for i in range(len(data)):
        x = data[i]
        while True:
            old_x = x
            in_bandwidth = []
            for j, point in enumerate(data):
                if euclidean_distance(x, point) < bandwidth:
                    in_bandwidth.append(j)
            x = np.mean(data[in_bandwidth], axis=0)
            if euclidean_distance(x, old_x) < 1e-3:
                for idx in in_bandwidth:
                    labels[idx] = len(centroids)
                break
        centroids.append(x)
    return np.array(centroids), labels


list = []
for index in range(38):
    boxes = np.load('true.npy')
    boxes = boxes[:, -1, index]
    boxes = np.ravel(boxes)
    list.append(boxes)
list = np.squeeze(np.array(list))

box = np.load("test0.npy")

centroids, labels = mean_shift(box, bandwidth=1)

print("样本标签:", labels)
