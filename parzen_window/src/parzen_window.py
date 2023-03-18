import numpy as np

from src.window_function import set_window_function


class ParzenWindow(object):

    def __init__(self, window_func: str, width):
        self.winfunc = set_window_function(name=window_func, width=width)
        self.samples = None

    def fit(self, samples: np.ndarray):
        self.samples = samples.copy()

    def predict(self, targets: np.ndarray) -> np.ndarray:
        C = self.winfunc(targets=targets, samples=self.samples)
        return C.mean(-1)
    

class ManifordParzenWindow(object):
    """考虑样本潜在流形结构的Parzen窗.

    论文 PDF: https://proceedings.neurips.cc/paper/2002/file/2d969e2cee8cfa07ce7ca0bb13c7a36d-Paper.pdf
    
    代码参考: https://github.com/tnybny/ManifoldParzenWindows

    参数:
        d: 主方向个数
        k: 邻居个数, 要求不小于`d`
        sig2: 正则化超参数
    """

    def __init__(self, d: int, k: int, sig2: float = np.square(0.09)):
        self.d = d
        self.k = k
        self.sig2 = sig2
        self.samples, self.V, self.lambda_vec = None, None, None

    def fit(self, samples: np.ndarray):
        l, n = samples.shape
        V = np.zeros((l, n, self.d))
        lambda_vec = np.zeros((l, self.d))

        for i in range(l):
            sample_i = samples[i]
            dist = np.zeros((l,))
            for j in range(l):
                if j == i:
                    dist[j] = 1e15
                    continue
                sample_j = samples[j]
                dist[j] = np.linalg.norm(sample_i - sample_j, 2)
            topk_inds = np.argsort(dist)[:self.k]
            M = np.zeros((self.k, n))
            for j in range(self.k):
                assert not topk_inds[j] == i
                M[j, :] = samples[topk_inds[j]] - sample_i
            
            _, s, Vi = np.linalg.svd(M, full_matrices=False)
            s_d = s[0 : self.d]
            V_d = Vi[0 : self.d, :].T

            V[i, :, :] = V_d
            lambda_vec[i, :] = (np.square(s_d) / l) + self.sig2

        self.samples = samples.copy()
        self.V = V.copy()
        self.lambda_vec = lambda_vec.copy()

    def predict(self, targets: np.ndarray):
        preds = []
        for target in targets:
            s = 0
            l = self.samples.shape[0]
            for i in range(l):
                s = s + self._local_gaussian(target, self.samples[i], self.V[i], self.lambda_vec[i])
            preds.append(s / l)
        return np.array(preds)

    def _local_gaussian(
        self, 
        x: np.ndarray, 
        x_i: np.ndarray, 
        V_i: np.ndarray,
        lambda_i: np.ndarray,
    ):
        """
        参数:
            x: 测试样本, 形状为`(n,)`
            x_i: 训练样本, 形状为`(n,)`
            V_i: `d`个特征向量, 形状为`(n, d)`
            lambda_i: `d`个特征值, 形状为`(d,)`

        返回值:
            x 处的高斯密度
        """
        n = x.shape[0]
        r = self.d * np.log(2 * np.pi) \
            + np.sum(np.log(lambda_i + self.sig2)) \
            + (n - self.d) * np.log(self.sig2)
        q = (1 / self.sig2) * np.sum(np.square(x - x_i))
        for j in range(self.d):
            temp = ((1.0 / lambda_i[j]) - (1.0 / self.sig2)) * np.square(V_i[:, j].dot(x - x_i))
            q = q + temp
        return np.exp(-0.5 * (r + q))
    

class FastParzenWindow(object):
    """对密度估计过程有一定加速的Parzen窗.

    论文 PDF: https://www.cs.bham.ac.uk/~pxt/PAPERS/fast_pw.pdf
    
    参数:
        r: 划分训练数据的距离阈值
    """

    def __init__(self, r: float):
        self.r = r
        self.m, self.C, self.P = [], [], []
        self.C_det, self.C_inv = [], []

    def fit(self, samples: np.ndarray):
        print('[FastParzenWindow] Fitting...', end="")
        left_inds = set(range(samples.shape[0]))
        while len(left_inds) > 0:
            q = left_inds.pop()
            inds_j = [q]
            rmv_inds = set()
            for ind in left_inds:
                if np.linalg.norm(samples[ind] - samples[q], 2) <= self.r:
                    inds_j.append(ind)
                    rmv_inds.add(ind)
            left_inds = left_inds.difference(rmv_inds)
            mj = samples[inds_j].mean(0)
            Cj = np.zeros((samples[q].shape[0], samples[q].shape[0]))
            for ind in inds_j:
                diff = samples[ind] - mj
                Cj += diff.reshape(-1, 1).dot(diff.reshape(1, -1))
            Cj = Cj / len(inds_j) + 1e-6 * np.eye(Cj.shape[0])
            
            self.m.append(mj)
            self.C.append(Cj)
            self.P.append(len(inds_j) / samples.shape[0])
            self.C_det.append(np.linalg.det(Cj))
            self.C_inv.append(np.linalg.inv(Cj))

        print('Done. Num of centers:', len(self.m))
    
    def predict(self, targets: np.ndarray):
        d = targets.shape[-1]

        preds = np.zeros((targets.shape[0],))
        for mj, Pj, Cj_det, Cj_inv in zip(self.m, self.P, self.C_det, self.C_inv):
            for i in range(targets.shape[0]):
                diff = targets[i] - mj
                preds[i] = preds[i] + \
                           Pj * \
                           (1.0 / (np.sqrt((2 * np.pi)**d * Cj_det))) * \
                           np.exp( -0.5 * (diff @ Cj_inv @ diff.T) )
        return preds