from scipy.stats import multivariate_normal
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
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
    # pickle.dump(data_train, open('data/data_train.pkl', 'wb'))
    # pickle.dump(data_test, open('data/data_test.pkl', 'wb'))

    # 真实分布的可视化
    XX, YY = np.mgrid[-5:4:0.05, -5:4:0.05]
    Z0 = multivariate_normal.pdf(np.dstack([XX, YY]), mean=data0_mean, cov=data0_cov)
    Z1 = multivariate_normal.pdf(np.dstack([XX, YY]), mean=data1_mean, cov=data1_cov)
    # 真实分布的3D概率密度图
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(XX, YY, Z0, cmap='rainbow')
    ax3.plot_surface(XX, YY, Z1, cmap='rainbow')
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title('真实分布 3D概率密度图')
    plt.savefig('pictures/true_3D.png')
    plt.show()
    # 真实分布的等高线概率密度图
    plt.contour(XX, YY, Z0)
    plt.contour(XX, YY, Z1)
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title('真实分布 等高线概率密度图')
    plt.savefig('pictures/true_contour.png')
    plt.show()

    # 训练集样本可视化
    X0_train = X_train[y_train == 0]
    X1_train = X_train[y_train == 1]
    plt.scatter(X0_train[:, 0], X0_train[:, 1], linewidths=0, s=10, alpha=0.5, c='b')
    plt.scatter(X1_train[:, 0], X1_train[:, 1], linewidths=0, s=10, alpha=0.5, c='r')
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title('训练集样本')
    plt.savefig('pictures/scatter_train.png')
    plt.show()

    # 测试集样本可视化
    X0_test = X_test[y_test == 0]
    X1_test = X_test[y_test == 1]
    plt.scatter(X0_test[:, 0], X0_test[:, 1], linewidths=0, s=10, alpha=0.5, c='b')
    plt.scatter(X1_test[:, 0], X1_test[:, 1], linewidths=0, s=10, alpha=0.5, c='r')
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title('测试集样本')
    plt.savefig('pictures/scatter_test.png')
    plt.show()
