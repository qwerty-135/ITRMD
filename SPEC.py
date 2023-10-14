import numpy as np
from skfeature.function.similarity_based import SPEC
from sklearn.datasets import load_iris  # 利用iris数据作为演示数据集

# 载入数据集
iris = load_iris()
X, y = iris.data, iris.target
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
boxes3 = read_pickle("machine-1-1_test.pkl")
boxes3=np.squeeze(np.array(boxes3))
boxes4= read_pickle("msl_test.pkl")
boxes4=np.squeeze(np.array(boxes4))
list=[]
for index in range(38):

    boxes = np.load('true.npy')

    # boxes=np.ravel(boxes)

    # print(X[1:3,1:3])

    # print(boxes.shape)
    boxes = boxes[:, -1, index]
    # print(len(boxes))
    boxes = np.ravel(boxes)
    # print(len(boxes))
    list.append(boxes)
list=np.squeeze(np.array(list))
datas = np.load("test0.npy")
datas=datas.T
list=list.T
print(len(list))
# 选择前100个观测点作为训练集
# 剩下的50个观测点作为测试集
# skfeature中的SEPC方法直接适用于连续变量

train_set = X[0:100,:]
# test_set = X[100:,]
# train_y = y[0:100]

num_feature = 30 # 从原数据集中选择两个变量

score = SPEC.spec(boxes4,style=2) # 计算每一个变量的得分

print(score)
# score.sort()  # [1 2 3 4 6 8]
feature_index = np.argsort(-score)
print(feature_index)
# feature_index = np.sort(score) #依据变量得分选择变量
transformed_train = boxes4[:,feature_index[0:num_feature]] # 转换训练集
# print(transformed_train)
# print(train_set[:,[1, 0]])
# assert np.array_equal(transformed_train, train_set[:,[1, 0]])  # 其选择了第一个及第二个变量

# transformed_test = test_set[:,feature_index[0:num_feature]] # 转换测试集
# assert np.array_equal(transformed_test, test_set[:,[1, 0]]) # 其选择了第一个及第二个变量