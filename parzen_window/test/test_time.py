import os
import sys
import time

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append('.')
from src.parzen_window import ParzenWindow, FastParzenWindow


results_dir = os.path.join("results", "test_time")
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def generate_data(N: int = 300, d: int = 5):
    samples = np.random.multivariate_normal(
        mean=np.zeros((d,)),
        cov=np.eye(d),
        size=N
    )

    dist = stats.multivariate_normal(mean=np.zeros((d,)), cov=np.eye(d))
    probs = dist.pdf(samples)

    return samples, probs


def proc(pw, samples, targets):
    time_start = time.time()
    pw.fit(samples)
    preds = pw.predict(targets)
    time_cost = time.time() - time_start
    return preds, time_cost


def plot_and_save_fig(x, y, zs, labels, colors, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    ax.set_title(title)
    for z, label, color in zip(zs, labels, colors):
        ax.plot_wireframe(x, y, z, linewidth=2, label=label, color=color)
    ax.legend()
    ax.set_xlabel("#Train")
    ax.set_ylabel("#Test")
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    fig.savefig(os.path.join(results_dir, "test_time_{}.png".format(title)))


def test_time():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    d = 10
    Ns = np.linspace(100, 1000, 19, dtype=np.int32)
    Ms = np.linspace(100, 2000, 26, dtype=np.int32)
    pts_xy, pts_pw, pts_fpw = [], [], []
    for N in Ns:
        for M in Ms:
            print(f"\rN={N}, M={M}  ", end="")
            X_train, y_train = generate_data(N, d=d)
            X_test, y_test = generate_data(M, d=d)
            parzen_window = ParzenWindow(window_func="Gaussian", width=0.1 * np.eye(d))
            f_parzen_window = FastParzenWindow(r=5.0)
            preds_pw, time_cost_pw = proc(parzen_window, X_train, X_test)
            preds_fpw, time_cost_fpw = proc(f_parzen_window, X_train, X_test)
            err_pw = (np.abs(y_test - preds_pw) / y_test).mean()
            err_fpw = (np.abs(y_test - preds_fpw) / y_test).mean()
            pts_xy.append([N, M])
            pts_pw.append([time_cost_pw, err_pw])
            pts_fpw.append([time_cost_fpw, err_fpw])
    pts_xy = np.array(pts_xy)
    pts_pw = np.array(pts_pw)
    pts_fpw = np.array(pts_fpw)
    
    grid_shape = (Ns.shape[0], Ms.shape[0])
    plt_x = pts_xy[:, 0].reshape(grid_shape)
    plt_y = pts_xy[:, 1].reshape(grid_shape)
    plt_t_pw = pts_pw[:, 0].reshape(grid_shape)
    plt_t_fpw = pts_fpw[:, 0].reshape(grid_shape)
    plt_e_pw = pts_pw[:, 1].reshape(grid_shape)
    plt_e_fpw = pts_fpw[:, 1].reshape(grid_shape)

    plot_and_save_fig(plt_x, plt_y, [plt_t_pw, plt_t_fpw], ["PW", "Fast PW"], ["b", "orange"], "Time Cost(sec)")
    plot_and_save_fig(plt_x, plt_y, [plt_e_pw, plt_e_fpw], ["PW", "Fast PW"], ["b", "orange"], "Error")
    print()


if __name__ == "__main__":
    test_time()
