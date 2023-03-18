import os
import re
import sys

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('.')
from src.parzen_window import ParzenWindow


results_dir = os.path.join("results", "test_sim_anomaly")
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 20


def read_data(path: str):
    data = np.load(path)

    gt = np.zeros_like(data)
    pattern = re.compile("\d+")
    anomaly_area = list(map(int, pattern.findall(os.path.split(path)[-1])))
    anomaly_area = np.asarray(anomaly_area, dtype=np.int64).reshape((-1, 4))
    anomaly_area_haxis = anomaly_area[:, [1, 3]]
    anomaly_area_vaxis = anomaly_area[:, [0, 2]]
    anomaly_area_haxis[:, 1] += anomaly_area_haxis[:, 0]
    anomaly_area_vaxis[:, 1] += anomaly_area_vaxis[:, 0]
    for hs, vs in zip(anomaly_area_haxis, anomaly_area_vaxis):
        hl, hr = hs[0], hs[1]
        vt, vb = vs[0], vs[1]
        for v in range(vt, vb):
            for h in range(hl, hr):
                gt[v, h] = 1

    return data, gt


def plot_two_img(img1, img2, span_width=10, title="", save_name="test_sim_anomaly"):
    plt_img = np.ones((img1.shape[0], img1.shape[1] + img2.shape[1] + span_width), dtype=np.uint8) * 255
    plt_img[:, :img1.shape[1]] = img1
    plt_img[:, (img1.shape[1] + span_width):] = img2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(plt_img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(os.path.join(results_dir, save_name + ".png"))


def detect(data, thres=0.9):
    pts = []
    for i in range(-1, data.shape[0] + 1):
        for j in range(-1, data.shape[1] + 1):
            if 0 <= i < data.shape[0] and 0 <= j < data.shape[1]:
                pts.append([i, j, *data[i, j].reshape(-1).tolist()])
            else:
                ti = min(max(0, i), data.shape[0] - 1)
                tj = min(max(0, j), data.shape[1] - 1)
                pts.append([i, j, *data[ti, tj].reshape(-1).tolist()])
    pts = np.array(pts)

    pw = ParzenWindow(window_func="Cube", width=2.0)
    pw.fit(pts[:, 0:])
    pred = pw.predict(pts[:, 0:])
    pred = pred.reshape((data.shape[0] + 2, data.shape[1] + 2))[1:-1, 1:-1]
    pred = pred / pred.max()
    pred[pred < thres] = 0
    pred[pred >= thres] = 1
    return 1 - pred


def calc_auc(pred, gt):
    pos = [i for i in range(len(gt)) if gt[i] == 1]
    neg = [i for i in range(len(gt)) if gt[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pred[i] > pred[j]:
                auc += 1
            elif pred[i] == pred[j]:
                auc += 0.5
    return auc / (len(pos) * len(neg))


def test_sim_anomaly():
    DATA_PATH = [
        "data/guassion_single_anomaly/10-20-25-41.npy",
        "data/guassion_single_anomaly/26-28-22-31.npy",
        "data/guassion_multi_overlap_anomaly/--8-10-45-9--10-24-30-11--5-10-28-21--21-20-18-30.npy",
        "data/guassion_multi_overlap_anomaly/--9-30-39-28--17-8-42-45.npy",
        "data/guassion_multi_overlap_anomaly/--11-29-30-6--9-8-32-18--19-15-16-4.npy",
        "data/guassion_multi_overlap_anomaly/--19-7-0-7--24-23-9-17--8-9-28-32--27-27-14-23.npy",
        "data/guassion_multi_overlap_anomaly/--24-19-4-27--2-28-10-14--4-17-22-3.npy",
    ]

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    for case_i, path in enumerate(DATA_PATH):
        data, gt = read_data(path)
        pred = detect(data, thres=0.9)
        auc = calc_auc(pred.reshape(-1), gt.reshape(-1))

        data = (data - data.min()) / (data.max() - data.min())
        data = np.array(data * 255.0, dtype=np.uint8)
        gt = np.array(gt * 255.0, dtype=np.uint8)
        pred = np.array(pred * 255.0, dtype=np.uint8)

        plot_two_img(data, pred, span_width=5, title=f"AUC: {auc:.4f}", 
                     save_name=f"test_sim_anomaly_case{case_i}")


if __name__ == "__main__":
    test_sim_anomaly()