import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from src.parzen_window import ParzenWindow, ManifordParzenWindow


results_dir = os.path.join("results", "test_manifold")
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def plot_and_save_2d_density(X, x_grid, y_grid, densities, extent, fig_title=""):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    point_size = 1

    ax.set_title(fig_title)
    densities[densities < 1.0] = 0.0
    ax.contour(x_grid, y_grid, densities, levels=5)
    # ax.imshow(densities, extent=extent, origin='lower')
    # fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax)
    
    ax.scatter(X[:, 0], X[:, 1], c="b", s=point_size)
    ax.set_aspect('equal', 'datalim')
    
    fig.savefig(os.path.join(results_dir, "test_manifold_{}.png".format(fig_title)))


def generate_data(N: int = 300):
    t = np.random.uniform(3, 15, N)

    samples = np.zeros((N, 2))
    samples[:, 0] = 0.04 * t * np.sin(t) + np.random.normal(0, 0.01, N)
    samples[:, 1] = 0.04 * t * np.cos(t) + np.random.normal(0, 0.01, N)

    return samples


def test_manifold():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    manifold_dimensions = 1
    num_neighbors = 11
    noise = np.square(0.01)
    samples = generate_data(N=300)

    parzen_window = ParzenWindow(window_func="Gaussian", width=0.0173**2 * np.eye(2))
    m_parzen_window = ManifordParzenWindow(d=manifold_dimensions, k=num_neighbors, sig2=noise)
    parzen_window.fit(samples)
    m_parzen_window.fit(samples)

    LEFT = -.7
    RIGHT = .7
    BOTTOM = -.6
    TOP = .6
    extent = [LEFT, RIGHT, BOTTOM, TOP]
    xs = np.arange(LEFT, RIGHT, 0.01)
    ys = np.arange(BOTTOM, TOP, 0.01)
    xx, yy = np.meshgrid(xs, ys)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    background = np.vstack((x_flat, y_flat)).T
    
    parzen_win_preds = parzen_window.predict(background)
    m_parzen_win_preds = m_parzen_window.predict(background)
    parzen_win_preds = parzen_win_preds.reshape(xx.shape)
    m_parzen_win_preds = m_parzen_win_preds.reshape(xx.shape)

    plot_and_save_2d_density(samples, xx, yy, parzen_win_preds, extent, "Parzen Window Prediction")
    plot_and_save_2d_density(samples, xx, yy, m_parzen_win_preds, extent, "Manifold Parzen Window Prediction")


if __name__ == "__main__":
    test_manifold()
