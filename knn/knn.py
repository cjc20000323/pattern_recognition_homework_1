import math
import numpy as np


class KNN:
    """
    处理2维数据的KNN
    """

    def __init__(self, K=None):
        self.N = None
        self.X = None
        self.K = K

    def fit(self, X):
        """
        传入数据
        :param X: (batch, feature_num)
        :return: None
        """
        self.X = X
        self.N = X.shape[0]
        if self.K is None:
            self.K = int(math.sqrt(self.N))

    def predict(self, X):
        """
        预测概率密度
        :param X: ([batch_dims], feature_num)
        :return: 概率密度数组: ([batch_dims], )
        """
        # 增加一个维度，这样可以利用numpy的广播机制进行后续处理
        X = np.expand_dims(X, axis=-2)  # ([batch_dims], 1, feature_num)
        # 距离的平方
        dist_2 = np.sum((X - self.X) ** 2, axis=-1)  # ([batch_dims], batch_self_X)
        # 沿着 batch_self_X 轴排序
        dist_2 = np.sort(dist_2)  # ([batch_dims], batch_self_X)
        V = np.pi * dist_2[..., self.K]  # ([batch_dims])
        P = self.K / self.N / V  # ([batch_dims])
        return P
