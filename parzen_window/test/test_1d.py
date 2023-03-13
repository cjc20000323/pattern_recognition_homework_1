import os
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append('.')
from src.parzen_window import ParzenWindow


results_dir = os.path.join("results", "test_1d")


def test_standard_normal_1d(N: int = 1000, M: int = 50, 
                            window_func: str = "cube", 
                            width=1.0):
    samples = np.random.randn(N)
    targets = np.random.randn(M)
    targets = np.sort(targets)
    parzen_window = ParzenWindow(window_func=window_func, width=width)

    parzen_window.fit(samples)
    pred_probs = parzen_window.predict(targets)
    true_probs = stats.norm.pdf(targets, 0, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("test_standard_normal_1d")
    standard_normal_pdf_x = np.linspace(-4, 4, 1000)
    standard_normal_pdf_y = stats.norm.pdf(standard_normal_pdf_x, 0, 1) 
    ax.plot(standard_normal_pdf_x, standard_normal_pdf_y, color="b", label="pdf")
    ax.scatter(targets, true_probs, marker="o", color="g")
    ax.plot(targets, pred_probs, color="r", label="pred")
    ax.legend()
    fig.savefig(os.path.join(results_dir, "test_standard_normal_1d.png"))


def main():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    test_standard_normal_1d()


if __name__ == "__main__":
    main()