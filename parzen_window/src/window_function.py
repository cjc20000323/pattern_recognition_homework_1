from typing import Union
import numpy as np


def _check_shape(x: np.ndarray):
    if x.ndim == 1:
        x = x.reshape((-1, 1))
    return x

class WindowFunction(object):

    def __init__(self, width):
        self.width = width

    def __call__(self, targets: np.ndarray, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class Cube(WindowFunction):
    """方窗函数.
    
    参数:
        targets: 待估计点矩阵, 是形状为`(M, d)`的二维`np.ndarray`对象, 表示共`M`个待估计点且特征维度为`d`
        samples: 观测样本点矩阵，是形状为`(N, d)`的二维`np.ndarray`对象, 表示共`N`个样本点且特征维度为`d`
    返回值:
        一个`MxN`维矩阵`C`, 其中元素`C[i, j]`表示第`j`个观测样本对第`i`个待估计点处概率密度估计的贡献
    """

    def __init__(self, width: float):
        super().__init__(width)

    def __call__(self, targets: np.ndarray, samples: np.ndarray) -> np.ndarray:
        targets = _check_shape(targets)
        samples = _check_shape(samples)

        diff = np.abs(targets[:, np.newaxis, :] - samples[np.newaxis, :, :])
        d = samples.shape[-1]
        C = np.where((diff <= (self.width / 2)).all(-1), (1 / self.width**d), 0)
        return C
    

class Gaussian(WindowFunction):
    """高斯窗函数.
    
    参数:
        targets: 待估计点矩阵, 是形状为`(M, d)`的二维`np.ndarray`对象, 表示共`M`个待估计点且特征维度为`d`
        samples: 观测样本点矩阵，是形状为`(N, d)`的二维`np.ndarray`对象, 表示共`N`个样本点且特征维度为`d`
    返回值:
        一个`MxN`维矩阵`C`, 其中元素`C[i, j]`表示第`j`个观测样本对第`i`个待估计点处概率密度估计的贡献
    """

    def __init__(self, width: Union[float, np.ndarray]):
        if type(width) is float:
            width = np.array([[width]])
        super().__init__(width)
        self.inv_width = np.linalg.inv(width)
        self.d = width.shape[0]
        self.c = 1 / np.sqrt( (2 * np.pi)**self.d * np.linalg.det(width) )

    def __call__(self, targets: np.ndarray, samples: np.ndarray) -> np.ndarray:
        targets = _check_shape(targets)
        samples = _check_shape(samples)
        M, N = targets.shape[0], samples.shape[0]

        diff = targets[:, np.newaxis, :] - samples[np.newaxis, :, :]
        dist = np.zeros((M, N))
        for i in range(M):
            dist[i, :] = np.diag(diff[i] @ self.inv_width @ diff[i].T)
            
        # for i in range(M):
        #     for j in range(N):
        #         dist[i, j] = diff[i, j] @ self.inv_width @ diff[i, j].T

        # diff = diff.reshape(-1, self.d)
        # dist = np.diag(diff @ self.inv_width @ diff.T).reshape(M, N)

        C = self.c * np.exp(-0.5 * dist)
        return C


def set_window_function(name: str, width, **kwargs):
    if name == "Cube":
        return Cube(width=width)
    elif name == "Gaussian":
        return Gaussian(width=width)
    else:
        raise ValueError('Unknown window function "{}"'.format(name))
