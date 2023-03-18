import os
import sys
from typing import List

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append('.')
from src.parzen_window import ParzenWindow


results_dir = os.path.join("results", "test_1d")
plt.rcParams['font.sans-serif'] = 'Times New Roman'


class GaussianDist1D(object):
    N_SPAN = 10000

    def __init__(
        self,
        weights: List[float] = [1.0],
        locs: List[float] = [0.],
        scales: List[float] = [1.0]
    ):
        self.weights = np.array(weights)
        self.locs = np.array(locs)
        self.scales = np.array(scales)
        self.x_l = np.min(self.locs - 3.5 * self.scales)
        self.x_r = np.max(self.locs + 3.5 * self.scales)
        self.pdf_xs, self.pdf_ys = self._sample(with_noise=False)
    
    def generate(self, N: int, is_train: bool = True):

        if not is_train:
            return np.linspace(self.x_l, self.x_r, N)

        sample_xs, sample_ys = self._sample(xs=self.pdf_xs.copy())

        inds = np.arange(N)
        sample_p = sample_ys / sample_ys.sum()
        inds = np.random.choice(len(sample_xs), size=N, p=sample_p)
        xs, ys = sample_xs[inds], sample_ys[inds]
        xs = xs.reshape(-1, 1)
        return xs, ys
    
    def _sample(self, xs=None, with_noise: bool = True):
        span = 1.0 * (self.x_r - self.x_l) / GaussianDist1D.N_SPAN
        if xs is None:
            xs = np.linspace(self.x_l, self.x_r, GaussianDist1D.N_SPAN)
        if with_noise:
            xs = xs + (np.random.uniform(size=GaussianDist1D.N_SPAN) - 0.5 ) * span
        ys = []
        for x in xs:
            prob = 0.0
            for weight, loc, scale in zip(self.weights, self.locs, self.scales):
                prob += weight * stats.norm.pdf(x, loc, scale)
            ys.append(prob)
        ys = np.array(ys)
        return xs, ys


def test_normal_1d_width(N: int = 1000, M: int = 500,
                         weights: List[float] = [1.0],
                         locs: List[float] = [0.],
                         scales: List[float] = [1.0],
                         window_func: str = "Cube",
                         widths=[1.0],
                         fig_title: str = ""):
    data_generator = GaussianDist1D(weights, locs,scales)
    train_xs, _ = data_generator.generate(N, is_train=True)
    test_xs = data_generator.generate(M, is_train=False)

    for width in widths:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.fill(data_generator.pdf_xs, data_generator.pdf_ys, ec='gray', fc='gray', alpha=0.3, label="PDF")
        ax.scatter(train_xs, np.zeros_like(train_xs), marker="|", label="Data", zorder=9, color="r")
        ax.set_title("N = {}, Width = {}".format(N, width))
        parzen_window = ParzenWindow(window_func=window_func, width=width)
        parzen_window.fit(train_xs)
        pred_ys = parzen_window.predict(test_xs)
        ax.plot(test_xs, pred_ys, label=f"pred", color="#ff7f0e", linewidth=3.0, alpha=0.5)
        ax.legend()
        
        fig.savefig(os.path.join(results_dir, "{}_{}_N={}_width={}.png".format(fig_title, window_func, N, width)))


def test_normal_1d_N(Ns: List[int] = [1000], 
                     M: int = 500,
                     weights: List[float] = [1.0],
                     locs: List[float] = [0.],
                     scales: List[float] = [1.0],
                     window_func: str = "Cube",
                     width=1.0,
                     fig_title: str = ""):
    data_generator = GaussianDist1D(weights, locs,scales)
    test_xs = data_generator.generate(M, is_train=False)

    for N in Ns:
        train_xs, _ = data_generator.generate(N, is_train=True)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.fill(data_generator.pdf_xs, data_generator.pdf_ys, ec='gray', fc='gray', alpha=0.3, label="PDF")
        ax.scatter(train_xs, np.zeros_like(train_xs), marker="|", label="Data", zorder=9, color="r")
        ax.set_title("N = {}, Width = {}".format(N, width))
        parzen_window = ParzenWindow(window_func=window_func, width=width)
        parzen_window.fit(train_xs)
        pred_ys = parzen_window.predict(test_xs)
        ax.plot(test_xs, pred_ys, label=f"pred", color="#ff7f0e", linewidth=3.0, alpha=0.5)
        ax.legend()
        
        fig.savefig(os.path.join(results_dir, "{}_{}_N={}_width={}.png".format(fig_title, window_func, N, width)))


def test_1d():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    test_normal_1d_width(weights=[0.2, 0.5, 0.3],
                         locs=[-5.5, 0.0, 6.0],
                         scales=[1.2, 1.5, 1.0],
                         window_func="Cube",
                         widths=[0.5, 1.0, 2.0, 4.0],
                         fig_title="test_1d")
    
    test_normal_1d_width(weights=[0.2, 0.5, 0.3],
                         locs=[-5.5, 0.0, 6.0],
                         scales=[1.2, 1.5, 1.0],
                         window_func="Gaussian",
                         widths=[0.1, 0.3, 0.9, 2.7],
                         fig_title="test_1d")
    
    test_normal_1d_N(Ns=[30, 270, 2430, 10000],
                     weights=[0.2, 0.5, 0.3],
                     locs=[-5.5, 0.0, 6.0],
                     scales=[1.2, 1.5, 1.0],
                     window_func="Cube",
                     width=2.0,
                     fig_title="test_1d")
    
    test_normal_1d_N(Ns=[30, 270, 2430, 10000],
                     weights=[0.2, 0.5, 0.3],
                     locs=[-5.5, 0.0, 6.0],
                     scales=[1.2, 1.5, 1.0],
                     window_func="Gaussian",
                     width=0.3,
                     fig_title="test_1d")


if __name__ == "__main__":
    test_1d()