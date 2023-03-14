from scipy.stats import multivariate_normal
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
"""
生成2元正态分布的随机数据，划分训练集和测试集，并保存数据到文件中
"""

data0_size = 6000  # 类型0的数据个数
data1_size = 4000  # 类型1的数据个数
train_size = 0.8  # 训练集划分比例
data0_mean = [1, 1]  # 类型0的均值
data0_cov = [[1, 0],  # 类型0的协方差
             [0, 1]]
data1_mean = [-1, -2]  # 类型1的均值
data1_cov = [[2, 0],  # 类型1的协方差
             [0, 1]]

if __name__ == '__main__':
    # 生成数据和标签
    X0 = multivariate_normal.rvs(mean=data0_mean, cov=data0_cov, size=data0_size, random_state=0)
    X1 = multivariate_normal.rvs(mean=data1_mean, cov=data1_cov, size=data1_size, random_state=0)
    y0 = np.array([0] * data0_size)
    y1 = np.array([1] * data1_size)

    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=0)
    data_train = {'X': X_train, 'y': y_train}
    data_test = {'X': X_test, 'y': y_test}

    # 保存数据到文件
    pickle.dump(data_train, open('data/data_train.pkl', 'wb'))
    pickle.dump(data_test, open('data/data_test.pkl', 'wb'))
