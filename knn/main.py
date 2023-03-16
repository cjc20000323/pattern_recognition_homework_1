import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from knn import KNN
plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

f_train = open('data/data_train.pkl', 'rb')
f_test = open('data/data_test.pkl', 'rb')

data_train = pickle.load(f_train)
data_test = pickle.load(f_test)

if __name__ == '__main__':
    data_train_np_X = np.array(data_train['X'])
    data_train_np_y = np.array(data_train['y'])
    data_train_np_X_0 = data_train_np_X[np.where(data_train_np_y == 0)]
    # data_train_np_y_0 = data_train_np_y[np.where(data_train_np_y == 0)]
    data_train_np_X_1 = data_train_np_X[np.where(data_train_np_y == 1)]
    # data_train_np_y_1 = data_train_np_y[np.where(data_train_np_y == 1)]

    data_test_np_X = np.array(data_test['X'])
    data_test_np_y = np.array(data_test['y'])
    data_test_np_X_0 = data_test_np_X[np.where(data_test_np_y == 0)]
    # data_test_np_y_0 = data_test_np_y[np.where(data_test_np_y == 0)]
    data_test_np_X_1 = data_test_np_X[np.where(data_test_np_y == 1)]
    # data_test_np_y_1 = data_test_np_y[np.where(data_test_np_y == 1)]

    # 先验概率
    prior_0 = (data_train_np_y == 0).sum() / len(data_train_np_X)
    prior_1 = (data_train_np_y == 1).sum() / len(data_train_np_X)

    knn_0 = KNN(K=150)
    knn_1 = KNN(K=150)
    knn_0.fit(data_train_np_X_0)
    knn_1.fit(data_train_np_X_1)

    condition_0 = knn_0.predict(data_test_np_X)
    condition_1 = knn_1.predict(data_test_np_X)

    predict = np.zeros_like(data_test_np_y)
    predict[prior_0 * condition_0 < prior_1 * condition_1] = 1
    acc = accuracy_score(data_test['y'], predict)
    print(f'Acc: {acc:.2%}')

    X, Y = np.mgrid[-5:4:0.05, -5:4:0.05]
    # KNN的3D概率密度图
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    Z_0 = knn_0.predict(np.dstack([X, Y]))
    Z_1 = knn_1.predict(np.dstack([X, Y]))
    ax3.plot_surface(X, Y, Z_0, cmap='rainbow')
    ax3.plot_surface(X, Y, Z_1, cmap='rainbow')
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title('KNN 3D概率密度图')
    plt.savefig('pictures/knn_3D.png')
    plt.show()
    # KNN的等高线概率密度图
    plt.contour(X, Y, Z_0)
    plt.contour(X, Y, Z_1)
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title('KNN 等高线概率密度图')
    plt.savefig('pictures/knn_contour.png')
    plt.show()
    # 测试集样本+决策面 的图
    condition_0 = knn_0.predict(np.dstack([X, Y]))
    condition_1 = knn_1.predict(np.dstack([X, Y]))
    pred = np.zeros_like(X)
    pred[prior_0 * condition_0 < prior_1 * condition_1] = 1
    plt.pcolormesh(X, Y, pred, cmap='bwr', alpha=0.5)
    plt.scatter(data_test_np_X_0[:, 0], data_test_np_X_0[:, 1], linewidths=0, s=10, alpha=1, c='b')
    plt.scatter(data_test_np_X_1[:, 0], data_test_np_X_1[:, 1], linewidths=0, s=10, alpha=1, c='r')
    plt.xlim(-5, 4)
    plt.ylim(-5, 4)
    plt.title(f'贝叶斯决策面 Acc:{acc:.2%}')
    plt.savefig('pictures/decision_surface.png')
    plt.show()
