from typing import List
import os
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sys.path.append('.')
from src.parzen_window import ParzenWindow


results_dir = os.path.join("results", "test_2d")
plt.rcParams['font.sans-serif'] = 'Times New Roman'

class GaussianDist2D(object):
    N_SPAN_X = 300
    N_SPAN_Y = 300

    def __init__(
        self,
        weights: List[float] = [1.0],
        means: List[np.ndarray] = [np.array([0, 0])],
        covs: List[np.ndarray] = [np.array([[1, 0], [0, 1]])]
    ):
        self.weights = np.array(weights)
        self.means = means
        self.covs = covs
        self.dists = []
        self.x_l, self.y_l = 1e9, 1e9
        self.x_r, self.y_r = -1e9, -1e9

        for mean, cov in zip(self.means, self.covs):
            dist = stats.multivariate_normal(mean=mean, cov=cov)
            self.dists.append(dist)

            x_mean, y_mean = mean[0], mean[1]
            x_var, y_var = cov[0, 0], cov[1, 1]
            x_l, x_r = x_mean - 3.5 * x_var, x_mean + 3.5 * x_var
            y_l, y_r = y_mean - 3.5 * y_var, y_mean + 3.5 * y_var
            self.x_l = min(self.x_l, x_l)
            self.y_l = min(self.y_l, y_l)
            self.x_r = max(self.x_r, x_r)
            self.y_r = max(self.y_r, y_r)

        self.pdf_xs, self.pdf_ys, self.pdf_pts = self._sample(with_noise=False)
    
    def generate(self, N: int, is_train: bool = True):

        if not is_train:
            xs = np.linspace(self.x_l, self.x_r, N)
            ys = np.linspace(self.y_l, self.y_r, N)
            pts = []
            for x in xs:
                for y in ys:
                    prob = 0.0
                    for weight, dist in zip(self.weights, self.dists):
                        prob += weight * dist.pdf(np.array([x, y]))
                    pts.append([x, y, prob])
            pts = np.array(pts)
            return pts

        _, _, sample_pts = self._sample(xs=self.pdf_xs.copy(), ys=self.pdf_ys.copy())

        inds = np.arange(N)
        sample_p = sample_pts[:, 2] / sample_pts[:, 2].sum()
        inds = np.random.choice(len(sample_pts), size=N, p=sample_p)
        pts = sample_pts[inds]
        return pts
    
    def _sample(self, xs=None, ys=None, with_noise: bool = True):
        x_span = 1.0 * (self.x_r - self.x_l) / GaussianDist2D.N_SPAN_X
        y_span = 1.0 * (self.y_r - self.y_l) / GaussianDist2D.N_SPAN_Y
        if xs is None:
            xs = np.linspace(self.x_l, self.x_r, GaussianDist2D.N_SPAN_X)
        if ys is None:
            ys = np.linspace(self.y_l, self.y_r, GaussianDist2D.N_SPAN_Y)
        if with_noise:
            xs = xs + (np.random.uniform(size=GaussianDist2D.N_SPAN_X) - 0.5) * x_span
            ys = ys + (np.random.uniform(size=GaussianDist2D.N_SPAN_Y) - 0.5) * y_span
        pts = []
        for x in xs:
            for y in ys:
                prob = 0.0
                for weight, dist in zip(self.weights, self.dists):
                    prob += weight * dist.pdf(np.array([x, y]))
                pts.append([x, y, prob])
        pts = np.array(pts)
        return xs, ys, pts


def test_normal_2d_width(N: int = 50 * 50, M: int = 30,
                         weights: List[float] = [1.0],
                         means: List[np.ndarray] = [np.array([0, 0])],
                         covs: List[np.ndarray] = [np.array([[1, 0], [0, 1]])],
                         window_func: str = "Cube",
                         widths=[1.0],
                         fig_title: str = ""):
    data_generator = GaussianDist2D(weights, means, covs)
    train_pts = data_generator.generate(N, is_train=True)
    test_pts = data_generator.generate(M, is_train=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    ax.set_title("GT")

    pdf_shape = (data_generator.pdf_ys.shape[0], data_generator.pdf_xs.shape[0])
    pdf_xs = data_generator.pdf_pts[:, 0].reshape(pdf_shape)
    pdf_ys = data_generator.pdf_pts[:, 1].reshape(pdf_shape)
    pdf_zs = data_generator.pdf_pts[:, 2].reshape(pdf_shape)
    ax.plot_surface(pdf_xs, pdf_ys, pdf_zs,
                    # color="gray",
                    cmap=cm.Oranges,
                    alpha=0.8, zorder=1)
    # ax.scatter(train_pts[:, 0], train_pts[:, 1], np.zeros_like(train_pts[:, 2]) - 0.03,
    #            marker="x", color="grey", label="Data", zorder=9)
    ax.scatter([0], [0], [-0.03], s=0)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    fig.savefig(os.path.join(results_dir, "{}_{}_GT.png".format(fig_title, window_func)))

    for width in widths:
        if window_func == "Gaussian":
            parzen_window = ParzenWindow(window_func=window_func, width=width * np.eye(2))
        else:
            parzen_window = ParzenWindow(window_func=window_func, width=width)
        parzen_window.fit(train_pts[:, :2])
        pred_probs = parzen_window.predict(test_pts[:, :2])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
        ax.set_title("N = {}, Width = {}".format(N, width))
        ax.scatter(train_pts[:, 0], train_pts[:, 1], np.zeros_like(train_pts[:, 2]) - 0.03,
                   marker="x", color="grey", label="Data", zorder=9)
        
        test_shape = (M, M)
        test_xs = test_pts[:, 0].reshape(test_shape)
        test_ys = test_pts[:, 1].reshape(test_shape)
        pred_probs = pred_probs.reshape(test_shape)
        ax.plot_wireframe(test_xs, test_ys, pred_probs, color="#ff7f0e", linewidth=0.5)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)

        ax.legend()
        # plt.show()
        fig.savefig(os.path.join(results_dir, "{}_{}_N={}_width={}.png".format(fig_title, window_func, N, width)))


def test_normal_2d_N(Ns: List[int] = [50 * 50], 
                     M: int = 50,
                     weights: List[float] = [1.0],
                     means: List[np.ndarray] = [np.array([0, 0])],
                     covs: List[np.ndarray] = [np.array([[1, 0], [0, 1]])],
                     window_func: str = "Cube",
                     width=1.0,
                     fig_title: str = ""):
    data_generator = GaussianDist2D(weights, means, covs)
    test_pts = data_generator.generate(M, is_train=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    ax.set_title("GT")

    pdf_shape = (data_generator.pdf_ys.shape[0], data_generator.pdf_xs.shape[0])
    pdf_xs = data_generator.pdf_pts[:, 0].reshape(pdf_shape)
    pdf_ys = data_generator.pdf_pts[:, 1].reshape(pdf_shape)
    pdf_zs = data_generator.pdf_pts[:, 2].reshape(pdf_shape)
    ax.plot_surface(pdf_xs, pdf_ys, pdf_zs,
                    # color="gray",
                    cmap=cm.Oranges,
                    alpha=0.8, zorder=1)
    # ax.scatter(train_pts[:, 0], train_pts[:, 1], np.zeros_like(train_pts[:, 2]) - 0.03,
    #            marker="x", color="grey", label="Data", zorder=9)
    ax.scatter([0], [0], [-0.03], s=0)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
    fig.savefig(os.path.join(results_dir, "{}_{}_GT.png".format(fig_title, window_func)))

    for N in Ns:
        train_pts = data_generator.generate(N, is_train=True)

        if window_func == "Gaussian":
            parzen_window = ParzenWindow(window_func=window_func, width=width * np.eye(2))
        else:
            parzen_window = ParzenWindow(window_func=window_func, width=width)
        parzen_window.fit(train_pts[:, :2])
        pred_probs = parzen_window.predict(test_pts[:, :2])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
        ax.set_title("N = {}, Width = {}".format(N, width))
        ax.scatter(train_pts[:, 0], train_pts[:, 1], np.zeros_like(train_pts[:, 2]) - 0.03,
                   marker="x", color="grey", label="Data", zorder=9)
        
        test_shape = (M, M)
        test_xs = test_pts[:, 0].reshape(test_shape)
        test_ys = test_pts[:, 1].reshape(test_shape)
        pred_probs = pred_probs.reshape(test_shape)
        ax.plot_wireframe(test_xs, test_ys, pred_probs, color="#ff7f0e", linewidth=0.5)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)

        ax.legend()
        # plt.show()
        fig.savefig(os.path.join(results_dir, "{}_{}_N={}_width={}.png".format(fig_title, window_func, N, width)))


def test_2d():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    test_normal_2d_width(weights=[0.4, 0.3, 0.3],
                         means=[
                            np.array([-3.5, 5.5]),
                            np.array([0.0, -3.0]),
                            np.array([0.5, 2.0]),
                         ],
                         covs = [
                            np.array([[2.0, 0.5], [0.5, 1.0]]),
                            np.array([[2.0, -0.5], [-0.5, 3.0]]),
                            np.array([[1.5, 0.0], [0.0, 2.0]])
                         ],
                         window_func="Cube", 
                         widths=[0.5, 1.0, 2.0, 4.0], 
                         fig_title="test_2d")
    
    test_normal_2d_width(weights=[0.4, 0.3, 0.3],
                         means=[
                            np.array([-3.5, 5.5]),
                            np.array([0.0, -3.0]),
                            np.array([0.5, 2.0]),
                         ],
                         covs = [
                            np.array([[2.0, 0.5], [0.5, 1.0]]),
                            np.array([[2.0, -0.5], [-0.5, 3.0]]),
                            np.array([[1.5, 0.0], [0.0, 2.0]])
                         ],
                         window_func="Gaussian", 
                         widths=[0.1, 0.3, 0.9, 2.7],
                         fig_title="test_2d")
    
    test_normal_2d_N(Ns=[6 * 6, 12 * 12, 24 * 24, 48 * 48],
                     weights=[0.4, 0.3, 0.3],
                     means=[
                        np.array([-3.5, 5.5]),
                        np.array([0.0, -3.0]),
                        np.array([0.5, 2.0]),
                     ],
                     covs = [
                        np.array([[2.0, 0.5], [0.5, 1.0]]),
                        np.array([[2.0, -0.5], [-0.5, 3.0]]),
                        np.array([[1.5, 0.0], [0.0, 2.0]])
                     ],
                     window_func="Cube", 
                     width=2.0,
                     fig_title="test_2d")
    
    test_normal_2d_N(Ns=[6 * 6, 12 * 12, 24 * 24, 48 * 48],
                     weights=[0.4, 0.3, 0.3],
                     means=[
                        np.array([-3.5, 5.5]),
                        np.array([0.0, -3.0]),
                        np.array([0.5, 2.0]),
                     ],
                     covs = [
                        np.array([[2.0, 0.5], [0.5, 1.0]]),
                        np.array([[2.0, -0.5], [-0.5, 3.0]]),
                        np.array([[1.5, 0.0], [0.0, 2.0]])
                     ],
                     window_func="Gaussian", 
                     width=0.3,
                     fig_title="test_2d")


if __name__ == "__main__":
    test_2d()