import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import os
import argparse

def uniform_background(low=0.0, high=1.0, dim1=1, dim2=64, dim3=64):
    background = np.random.uniform(low=low, high=high, size=(dim1,dim2,dim3))
    return background.squeeze()

def guassion_background(loc=0.0, scale=1.0, dim1=1, dim2=64, dim3=64):
    background = np.random.normal(loc=loc, scale=scale, size=(dim1,dim2,dim3))
    return background.squeeze()

def guassion_single_anomaly(background, loc=0.0, scale=1.0, save_anomaly = True, save_name = None, save_dir=None):
    anomaly_area_X = np.random.randint(2, background.shape[-2]/2 - 1) # anomaly area length
    anomaly_area_Y = np.random.randint(2, background.shape[-1]/2 - 1) # anomaly area height
    anomaly_point_X = np.random.randint(background.shape[-2] - anomaly_area_X) # anomaly area top left corner X
    anomaly_point_Y = np.random.randint(background.shape[-1] - anomaly_area_Y) # anomaly area top left corner Y
    foreground = np.random.normal(loc=loc, scale=scale, size=(anomaly_point_X, anomaly_point_Y)) # anomaly data
    background[anomaly_area_X:anomaly_area_X+anomaly_point_X, anomaly_area_Y:anomaly_area_Y+anomaly_point_Y] += foreground
    
    if save_name == None:
        save_name = str(anomaly_area_X) + "-" + str(anomaly_area_Y) + "-" + str(anomaly_point_X) + "-" + str(anomaly_point_Y) + ".npy"
    else:
        save_name = save_name + "-" + str(anomaly_area_X) + "-" + str(anomaly_area_Y) + "-" + str(anomaly_point_X) + "-" + str(anomaly_point_Y) + ".npy"
    
    if save_anomaly:
        if save_dir == None:
            np.save(file = save_name, arr = background)
        else:
            np.save(file = os.path.join(save_dir,save_name), arr = background)

    return background, save_name

def guassion_multi_nonoverlap_anomaly(background, loc=0.0, scale=1.0, save_anomaly = False, low=2, high=5):
    pass

def guassion_multi_overlap_anomaly(background, loc=0.0, scale=1.0, save_anomaly = True, low=2, high=5, save_name = None, save_dir=None):
    times = np.random.randint(low,high)
    save_names = ""
    for _ in range(times):
        background, name = guassion_single_anomaly(background, loc=loc, scale=scale, save_anomaly = False)
        save_names = save_names + "--" + name.split(".")[0]

    if save_name == None:
        save_name = save_names + ".npy"
    else:
        save_name = save_name + save_names + ".npy"
    if save_anomaly:      
        if save_dir == None:
            np.save(file = save_name, arr = background)
        else:
            np.save(file = os.path.join(save_dir,save_name), arr = background)

    return background, save_name

def heatmap_output(picture, savename=None, savedir=None):
    plt.figure()
    heatmap = sns.heatmap(picture).get_figure()
    if savename == None:
        savename = "heatmap.jpg"
    elif savename[-3:] != "jpg":
        savename += ".jpg"
    else:
        pass

    if savedir == None:
        heatmap.savefig(savename)
    else:
        heatmap.savefig(os.path.join(savedir,savename))

def test():
    # 获得均匀分布背景
    background = uniform_background()
    # 获得并保存单个高斯异常矩阵
    guass_anomaly, _ = guassion_single_anomaly(copy.deepcopy(background))
    # 获得并保存可重叠多个高斯异常矩阵
    guass_anomalys, _ = guassion_multi_overlap_anomaly(copy.deepcopy(background))
    # 显示并保存热力图
    heatmap_output(background, savename="heatmap_uniform.jpg")
    heatmap_output(guass_anomaly, savename="heatmap_gauss.jpg")
    heatmap_output(guass_anomalys, savename="heatmap_gausses.jpg")

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="./data_generation/data", type=str, help='')
    parser.add_argument('--output_name', default=None, type=str, help='')
    parser.add_argument('--generate_number','--gn', default=1, type=int, help='the data number scale')
    parser.add_argument('--generate_type','--gt', default="guassion_single_anomaly", choices=['guassion_single_anomaly', 'guassion_multi_overlap_anomaly'], type=str, help='the anomaly type')
    args = parser.parse_args()

    # test()


    save_dir = os.path.join(args.output_dir,args.generate_type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    background = uniform_background()
    for _ in range(args.generate_number):
        if args.generate_type == "guassion_single_anomaly":
            guass_anomaly, _ = guassion_single_anomaly(copy.deepcopy(background), save_dir=save_dir, save_name=args.output_name)
        elif args.generate_type == "guassion_multi_overlap_anomaly":
            guassion_multi_overlap_anomaly(copy.deepcopy(background), save_dir=save_dir, save_name=args.output_name)
        else:
            raise ValueError("undefined anomaly type")