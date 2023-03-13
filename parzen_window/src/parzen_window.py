import numpy as np

from src.window_function import set_window_function


class ParzenWindow(object):

    def __init__(self, window_func: str, width):
        self.winfunc = set_window_function(name=window_func, width=width)
        self.samples = None

    def fit(self, samples: np.ndarray):
        self.samples = samples.copy()

    def predict(self, targets: np.ndarray):
        C = self.winfunc(targets=targets, samples=self.samples)
        return C.mean(-1)