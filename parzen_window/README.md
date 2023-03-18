# Parzen窗

### **一、基本原理实现测试**

目前实现了方窗函数和高斯窗函数，主要观察了两种函数估计得到的概率密度函数的光滑性差异，以及不同窗宽、不同样本数对估计结果的影响。

#### **1.1 窗宽对估计结果的影响**

如图 1 所示是方窗函数在不同窗宽（width）下对某一密度函数估计的结果，可以看到窗宽越大，估计出的密度函数越光滑。

<div align="center">
    <img src="results\test_1d\test_1d_Cube_N=1000_width=0.5.png" width="49%">
    <img src="results\test_1d\test_1d_Cube_N=1000_width=1.0.png" width="49%">
    <img src="results\test_1d\test_1d_Cube_N=1000_width=2.0.png" width="49%">
    <img src="results\test_1d\test_1d_Cube_N=1000_width=4.0.png" width="49%">
</div>
<center><font size=2><b>图 1 方窗函数在不同窗宽下的估计结果</b></font></center>

<br/>

如图 2 所示是高斯窗函数在不同窗宽下对某一密度函数估计的结果，可以看到窗宽与估计结果光滑性同样是正相关的，同时高斯窗的估计要明显比方窗更光滑。

<div align="center">
    <img src="results\test_1d\test_1d_Gaussian_N=1000_width=0.1.png" width="49%">
    <img src="results\test_1d\test_1d_Gaussian_N=1000_width=0.3.png" width="49%">
    <img src="results\test_1d\test_1d_Gaussian_N=1000_width=0.9.png" width="49%">
    <img src="results\test_1d\test_1d_Gaussian_N=1000_width=2.7.png" width="49%">
</div>
<center><font size=2><b>图 2 高斯窗函数在不同窗宽下的估计结果</b></font></center>

如图 3、图 4 所示，对二维数据的分布进行估计时窗口大小也会影响结果的光滑性。

<br/>

<div align="center">
    <img src="results\test_2d\test_2d_Cube_GT.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=2500_width=0.5.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=2500_width=1.0.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=2500_width=2.0.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=2500_width=4.0.png" width="32%">
</div>
<center><font size=2><b>图 3 方窗函数在不同窗宽下对一组二维数据分布的估计结果</b></font></center>

<br/>
<div align="center">
    <img src="results\test_2d\test_2d_Gaussian_GT.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=2500_width=0.1.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=2500_width=0.3.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=2500_width=0.9.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=2500_width=2.7.png" width="32%">
</div>
<center><font size=2><b>图 4 高斯窗函数在不同窗宽下对一组二维数据分布的估计结果</b></font></center>

<br/>

#### **1.2 样本数对估计结果的影响**

总体上看是样本数越多估计越准确。图 5、图 6 分别是方窗函数和高斯窗函数在 1D 数据上的估计结果，图 7、图 8 分别是在 2D 数据上的估计结果。

<div align="center">
    <img src="results\test_1d\test_1d_Cube_N=30_width=2.0.png" width="49%">
    <img src="results\test_1d\test_1d_Cube_N=270_width=2.0.png" width="49%">
    <img src="results\test_1d\test_1d_Cube_N=2430_width=2.0.png" width="49%">
    <img src="results\test_1d\test_1d_Cube_N=10000_width=2.0.png" width="49%">
</div>
<center><font size=2><b>图 5 方窗函数在不同样本数下的估计结果</b></font></center>

<br/>

<div align="center">
    <img src="results\test_1d\test_1d_Gaussian_N=30_width=0.3.png" width="49%">
    <img src="results\test_1d\test_1d_Gaussian_N=270_width=0.3.png" width="49%">
    <img src="results\test_1d\test_1d_Gaussian_N=2430_width=0.3.png" width="49%">
    <img src="results\test_1d\test_1d_Gaussian_N=10000_width=0.3.png" width="49%">
</div>
<center><font size=2><b>图 6 高斯窗函数在不同样本数下的估计结果</b></font></center>

<br/>

<div align="center">
    <img src="results\test_2d\test_2d_Cube_GT.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=36_width=2.0.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=144_width=2.0.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=576_width=2.0.png" width="32%">
    <img src="results\test_2d\test_2d_Cube_N=2304_width=2.0.png" width="32%">
</div>
<center><font size=2><b>图 7 方窗函数在不同样本数下对一组二维数据分布的估计结果</b></font></center>

<br/>
<div align="center">
    <img src="results\test_2d\test_2d_Gaussian_GT.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=36_width=0.3.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=144_width=0.3.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=576_width=0.3.png" width="32%">
    <img src="results\test_2d\test_2d_Gaussian_N=2304_width=0.3.png" width="32%">
</div>
<center><font size=2><b>图 8 高斯窗函数在不同样本数下对一组二维数据分布的估计结果</b></font></center>

<br/>

### **二、简单改进**

#### **2.1 考虑数据的流形结构**

[文献](https://proceedings.neurips.cc/paper/2002/file/2d969e2cee8cfa07ce7ca0bb13c7a36d-Paper.pdf)提到了一般的高斯窗函数可能并不符合数据本身的流形结构，于是提出了一种名为 Manifold Parzen Window 的方法，该方法可自动捕获数据潜在的流形结构并进行 Parzen 窗计算。如图 9 所示是普通的 Parzen 窗（左子图）和该文献所提方法（右子图）对一人工数据的密度估计结果，可以看到其方法更好地识别出了数据的一维结构（更少的间断）。

<div align="center">
    <img src="results\test_manifold\test_manifold_Parzen Window Prediction.png" width="49%">
    <img src="results\test_manifold\test_manifold_Manifold Parzen Window Prediction.png" width="49%">
</div>
<center><font size=2><b>图 9 普通Parzen窗与Manifold Parzen窗估计结果对比</b></font></center>

<br/>

#### **2.2 估计过程加速**

按照 Parzen 窗的基本公式，估计时需要遍历所有的样本，这在数据量大的时候很耗费时间。[文献](https://www.cs.bham.ac.uk/~pxt/PAPERS/fast_pw.pdf)提出了一种名为 Fast Parzen Window 的加速估计的方法：先根据样本间距离进行聚类，随后在估计时仅取各个类簇的中心参与计算（而不是所有样本）。如图 10 所示比较了普通 Parzen 窗与该方法在不同训练样本数和测试样本数下的时间消耗，可以看到该方法在数据量较大时明显快于普通的 Parzen 窗；图 10 中还比较了两种方法的估计误差（右子图），可以看到该方法虽然没有去使用所有训练样本，但是总体的误差不劣于普通 Parzen 窗。

<div align="center">
    <img src="results\test_time\test_time_Time Cost(sec).png" width="49%">
    <img src="results\test_time\test_time_Error.png" width="49%">
</div>
<center><font size=2><b>图 10 普通Parzen窗与Fast Parzen窗估计结果对比</b></font></center>

<br/>

### **三、应用**

这一部分简单模拟了图像异常区域检测问题，即在一块均匀分布的背景上的某些部分添加高斯异常，要求将这些高斯异常找出来。处理思路是根据像素周边的灰度值分布情况来判断该像素自身是否是异常，即先根据像素邻居的灰度分布计算出该像素出现的概率（密度），最后由预先设置的阈值判断该像素是否是异常点。这个过程不需要事先指定哪种像素灰度值范围属于正常或异常。该图 11 给出了部分检测结果。

<div align="center">
    <img src="results\test_sim_anomaly\test_sim_anomaly_case0.png" width="49%">
    <img src="results\test_sim_anomaly\test_sim_anomaly_case1.png" width="49%">
    <img src="results\test_sim_anomaly\test_sim_anomaly_case2.png" width="49%">
    <img src="results\test_sim_anomaly\test_sim_anomaly_case3.png" width="49%">
    <img src="results\test_sim_anomaly\test_sim_anomaly_case4.png" width="49%">
    <img src="results\test_sim_anomaly\test_sim_anomaly_case5.png" width="49%">
</div>
<center><font size=2><b>图 11 使用 Parzen 窗对均匀分布背景中的高斯异常进行检测 </b></font></center>
