import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

f_train = open('data/data_train.pkl', 'rb')
f_test = open('data/data_test.pkl', 'rb')

data_train = pickle.load(f_train)
data_test = pickle.load(f_test)


def gauss_kernel(x, sigma):
    return 1 / math.sqrt(2 * math.pi) / sigma * math.exp(-1 / 2 * math.pow(x, 2) / math.pow(sigma, 2))


def knn(k, plot_loc, data, use_gauss=False):
    '''
    :param k: KNN算法中的k值，超参数
    :param plot_loc: 估计该位置的概率密度
    :param data: 两类坐标数据中的一类
    :param use_gauss: 是否使用高斯核
    :return: 估计位置的概率密度
    '''

    dis_list = []
    for (x, y) in data:
        dis = math.sqrt(math.pow(plot_loc[0] - x, 2) + math.pow(plot_loc[1] - y, 2))
        dis_list.append(dis)

    dis_list.sort()
    r = dis_list[k-1]  # 排序后，距离是从小到大的，这样，取第k个大小的，那么恰好选中了k个点，这里暂不考虑距离相同的多个点的问题
    n = len(dis_list)
    V = math.pi * math.pow(r, 2)
    count = 0
    if use_gauss:
        for i in range(k):
            count += gauss_kernel(dis_list[i] / r, 1.3)
    else:
        count = k

    p = count / n / V

    return p


if __name__ == '__main__':
    data_train_np_X = np.array(data_train['X'])
    data_train_np_y = np.array(data_train['y'])
    data_train_np_X_0 = data_train_np_X[np.where(data_train_np_y == 0)]
    data_train_np_y_0 = data_train_np_y[np.where(data_train_np_y == 0)]
    data_train_np_X_1 = data_train_np_X[np.where(data_train_np_y == 1)]
    data_train_np_y_1 = data_train_np_y[np.where(data_train_np_y == 1)]

    data_test_np_X = np.array(data_test['X'])
    data_test_np_y = np.array(data_test['y'])
    data_test_np_X_0 = data_test_np_X[np.where(data_test_np_y == 0)]
    data_test_np_y_0 = data_test_np_y[np.where(data_test_np_y == 0)]
    data_test_np_X_1 = data_test_np_X[np.where(data_test_np_y == 1)]
    data_test_np_y_1 = data_test_np_y[np.where(data_test_np_y == 1)]

    prior_0 = 0.6
    prior_1 = 0.4

    predict = []
    k = 150
    for x, y in tqdm(data_test['X']):
        condition_0 = knn(k, [x, y], data_train_np_X_0)
        condition_1 = knn(k, [x, y], data_train_np_X_1)
        if prior_0 * condition_0 > prior_1 * condition_1:
            predict.append(0)
        else:
            predict.append(1)

    acc = accuracy_score(data_test['y'], predict)
    print(acc)

    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    xx = np.arange(-4, 4, 0.05)
    yy = np.arange(-4, 4, 0.05)
    X, Y = np.meshgrid(xx, yy)

    print(X.shape)
    print(Y.shape)
    Z_0 = np.zeros((160, 160), float)
    Z_1 = np.zeros((160, 160), float)
    for i in tqdm(range(len(X))):
        for j in range(len(X[0])):
            Z_0[i][j] = knn(k, [X[i][j], Y[i][j]], data_train_np_X_0)
            Z_1[i][j] = knn(k, [X[i][j], Y[i][j]], data_train_np_X_1)
            # Z_0[i][j] += Z_1[i][j]
    ax3.plot_surface(X, Y, Z_0, cmap='rainbow')
    ax3.plot_surface(X, Y, Z_1, cmap='rainbow')

    # plt.scatter(data_train_np_X_0[:, 0], data_train_np_X_0[:, 1], c='b')
    # plt.scatter(data_train_np_X_1[:, 0], data_train_np_X_1[:, 1], c='r')
    plt.show()

    plt.contour(X, Y, Z_0)
    plt.contour(X, Y, Z_1)
    plt.show()

