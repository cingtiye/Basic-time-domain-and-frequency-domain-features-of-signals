<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"> </script>
# 信号下的基本时域频域特征
星期日, 16. 六月 2019 07:41下午
最近忙于项目，因此本期博客就简单梳理一下信号(笔者做的多数情况下是三相电流数据)下的基本时域和频域特征。
**提取特征可以用tsfresh这个库来实现**，但是这里还是采用了numpy。
##  1 时域特征
假设共有m条数据，每一条数据长度为n，第ｉ条数据第ｊ个数据点用\\(z_{ij}\\)表示，并且z为数组，不能是列表，否则以下一些程序会报错。**以下主要从数学公式和python实现来叙述**。
### (1) 含量纲的时域特征
含量纲的时域特征，笔者简单整理出了十个，其中包括最大值(maximum)、最小值(minimum)、极差(range)、均值(mean)、中位数(media)、众数(mode)、标准差(standard deviation)、均方根值(root mean square/rms)、均方值(mean square/ms)、k阶中心/原点矩。
**导入所需要的库**

	import numpy as np
#### 最大值
$$max(z_i)$$

	max_z = np.max(z, axis=1)
#### 最小值
$$min(z_i)$$

	min_z = np.min(z, axis=1)
#### 极差
$$max(z_i)-min(z_i)$$

	range_z = np.max(z, axis=1)-np.min(z, axis=1)
#### 均值
$$\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}$$

	mean_z = np.mean(z, axis=1)
#### 中位数
将一组数从小到大排序，出现在中间的数(当n为奇数时)或者中间两个数的均值(当n为偶数时)

	media_z = np.median(z, axis=1)
#### 众数
一组数从大到小排序，出现次数最多的数(当有多个数出现次数一样，取最小的数)

	import scipy.stats
	mode_z = scipy.stats.mode(z, axis=1)[0].reshape([-1])
#### 标准差
$$\sqrt{\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^2}$$

	std_z = np.std(z, axis=1)
#### 均方根值
$$\sqrt{\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}^2}$$

	from sklearn.metrics import mean_squared_error
	rms_z = [np.sqrt(mean_squared_error(zi, np.zeros(len(zi)))) for zi in z]
	rms_z = np.array(rms_z)
#### 均方值(二阶中心矩)
$$\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}^2$$

	from sklearn.metrics import mean_squared_error
	ms_z = [mean_squared_error(zi, np.zeros(len(zi))) for zi in z]
	ms_z = np.array(ms_z)
#### ｋ阶中心矩/原点矩
##### k阶中心矩
$$\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^k$$
##### k阶原点矩
$$\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij})^k$$

	def k_order_moment(z, k, is_center=True, is_origin=True):
	    """
	    Calculate k-order center moment and k-order origin moment of z
	    :param z: array_like
	    :param k: int
	    :param is_center: bool; whether calculate k-order center moment
	    :param is_origin: bool; whether calculate k-order origin moment
	    :return: tuple; return k-order center moment and k-order origin moment
	    """
	    if (is_center is False) and (is_origin is False):
	        raise ValueError("At least one of is_center and is_origin is True")
	    if (type(k) is not int) or (k < 0):
	        raise TypeError("k must be a integrate and more than 0")
	    if type(z) is list:
	        z = np.array(z)

	    mean_z = np.mean(z, axis=1)
	    if is_origin is False:
	        k_center = np.mean([(z[i]-mean_z[i])**k for i in range(z.shape[0])], axis=1)
	        return (k_center, None)
	    if is_center is False:
	        k_origin = np.mean([z[i]**k for i in range(z.shape[0])], axis=1)
	        return (None, k_origin)
	    if is_center and is_origin:
	        k_center = np.mean([(z[i] - mean_z[i]) ** k for i in range(z.shape[0])], axis=1)
	        k_origin = np.mean([z[i] ** k for i in range(z.shape[0])], axis=1)
	        return (k_center, k_origin)
### (2) 无量纲的时域特征
无量纲的时域笔者主要列举了６个，分别为偏度(skewness)，峰度(kurtosis)，峰度因子(kurtosis factor)、波形因子(waveform factor)、脉冲因子(pulse factor)、裕度因子(margin factor)。
#### 偏度(三阶标准矩)
$$E[(\frac{z_{ij}-\mu}{\sigma})^3]$$
**\\(\mu,\sigma\\)为总体均值和标准差，不是样本均值和标准差!!!**
偏度可通过下面两种方法计算：
方法１：
$$\frac
{\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^3}
{[\frac{1}{n-1}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^2]^{3/2}}
$$
方法２：
$$
\frac{\sqrt{n(n-1)}}{n-2}
\left[\frac
{\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^3}
{[\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^2]^{3/2}}
\right]
$$
**python和大多数软件采用方法２求偏度**

	import pandas as pd
	skew_z = pd.DataFrame(z.transpose()).skew().values
#### 峰度(四阶标准矩)
$$E[(\frac{z_{ij}-\mu}{\sigma})^4]$$
**\\(\mu,\sigma\\)为总体均值和标准差，不是样本均值和标准差!!!**
峰度同偏度一样也有两种方法计算：
方法１：
$$\frac
{\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^4}
{[\frac{1}{n}\sum_{j=0}^{n-1}(z_{ij}-\overline{z_i})^2]^{2}}-3=
\frac{m_{4}}{m_{2}^2}-3
$$

方法２(**\\(n>3\\)**)：
$$
\frac
{n^2((n+1)m_4-3(n-1)m_2^2)}
{(n-1)(n-2)(n-3)}
\frac
{(n-1)^2}
{n^2m_2^2}
$$
**python和大多数软件采用方法２求峰度**

	import pandas as pd
	kurt_z = pd.DataFrame(z.transpose()).kurt().values
#### 峰度因子
$$\frac{max(z_i)}{\sqrt{\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}^2}}$$

	kurt_factor_z = max_z/rms_z
#### 波形因子
$$\frac{\sqrt{\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}^2}}{\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}}$$

	wave_factor_z = rms_z/mean_z
#### 脉冲因子
$$\frac{max(z_i)}{|\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}|}$$

	pulse_factor_z = max_z/abs(mean_z)
#### 裕度因子
$$\frac{max(z_i)}{\frac{1}{n}\sum_{j=0}^{n-1}z_{ij}^2}$$

	margin_factor_z = max_z/ms_z
