## 使用方法
  python --gn 10 --gt guassion_multi_overlap_anomaly
  
  args：
  --output_dir 指定输出文件夹
  --output_name 自定义输出命名
  --gn 生成异常样例个数
  --gt 生成异常样例类型 choices=['guassion_single_anomaly', 'guassion_multi_overlap_anomaly']




## 函数功能

### **uniform_background**：获得均匀分布背景

​	**para：**	low：最小值（default=0）；

​						high：最大值（default=1）；

​						dim1：输入第一维（default=1）；

​						dim2：输入第二维（default=64）；

​						dim2：输入第三维（default=64）；



### **guassion_background**：获得高斯分布背景

​	**para：**	loc：均值（default=0）；

​						scale：方差（default=1）；

​						dim1：输入第一维（default=1）；

​						dim2：输入第二维（default=64）；

​						dim2：输入第三维（default=64）；



### **guassion_single_anomaly**：在随机位置插入一个高斯异常

​	**para：**	background：背景分布

​						loc：均值（default=0）；

​						scale：方差（default=1）；

​						save_anomaly：是否保存（default=True）；

​						save_name：保存文件名（default=None）；（主要用于批量生成）

​						save_dir：保存文件夹路径（default=None）；

*npy文件默认保存名为（save_dir/save_name-长-高-左上角x坐标-左上角y坐标.npy)



### **guassion_multi_overlap_anomaly**：在随机位置插入2-5个高斯异常（不同异常之间可能存在重叠区域）

**para：**	background：背景分布

​						loc：均值（default=0）；

​						scale：方差（default=1）；

​						low：最小异常个数（default=2）；

​						high：最多异常个数（default=5）；

​						save_anomaly：是否保存（default=True）；

​						save_name：保存文件名（default=None）；（主要用于批量生成）

​						save_dir：保存文件夹路径（default=None）；

*npy文件默认保存名为（save_dir/save_name--长-高-左上角x坐标-左上角y坐标--长-高-左上角x坐标-左上角y坐标--.npy)



### **heatmap_output**：获得热力图输出

​						picture：热力矩阵

​						save_name：保存文件名（default=None）；（主要用于批量生成）

​						save_dir：保存文件夹路径（default=None）；
