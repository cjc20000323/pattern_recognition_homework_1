import numpy as np


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

    def __init__(self, width):
        super().__init__(width)

    def __call__(self, targets: np.ndarray, samples: np.ndarray) -> np.ndarray:
        targets = self._check_shape(targets)
        samples = self._check_shape(samples)

        diff = np.abs(targets[:, np.newaxis, :] - samples[np.newaxis, :, :])
        d = samples.shape[-1]
        C = np.where((diff <= (self.width / 2)).all(-1), (1 / self.width**d), 0)
        return C
    
    def _check_shape(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        return x
    

def set_window_function(name: str, width, **kwargs):
    if name == "cube":
        return Cube(width=width)
    else:
        raise ValueError('Unknown window function "{}"'.format(name))


if __name__ == '__main__':
    a = np.random.randint(0, 4, (6, 4)) * 1.0
    b = np.random.randint(0, 4, (9, 4)) * 1.0
    cube = set_window_function(name="cube", width=4.0)
    print(a)
    print()
    print(b)
    print()
    print(np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :]))
    print()
    res = cube(a, b)
    print(res)
    print(res.shape)
