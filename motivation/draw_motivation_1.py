import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline
from pylab import xticks,yticks

plt.style.use("seaborn-v0_8-darkgrid")
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 160)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth


# example data
x = np.linspace(1,16,16)
y = np.load('motivation/npy/y.npy')[0:16]
ystd = np.load('motivation/npy/yerr.npy')[0:16]
cost = np.load('motivation/npy/cost.npy')[0:16]
cost_std = np.load('motivation/npy/cost_std.npy')[0:16]
qos = np.load('motivation/npy/qos.npy')[0:16]
qos_std = np.load('motivation/npy/qos_std.npy')[0:16]

y[1] = 18.2475

def min_max_normalization(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

cost_normalized_std = cost_std/(np.max(cost)-np.min(cost))
cost_normalized = min_max_normalization(cost)

yerr_normalized = ystd/(np.max(y)-np.min(y))
y_normalized = min_max_normalization(1/y)

qos_normalized_std = qos_std/(np.max(qos)-np.min(qos))
qos_normalized = min_max_normalization(qos)

# xs,y_normalized = smooth_xy(x, y_normalized)
# xs,cost_normalized = smooth_xy(x, cost_normalized)
# xs,qos_normalized = smooth_xy(x, qos_normalized)

fig, ax = plt.subplots(figsize=(8, 7))
plt.plot(x, cost_normalized, c=(253/255,185/255,107/255),label='Cost Per-Frame',linewidth = 2, \
            clip_on=False, marker='o', markersize=7 ,markeredgecolor=(249/255, 172/255, 19/255),markeredgewidth=2)

plt.plot(x, qos_normalized,c=(254/255, 162/255, 158/255),label='Latency',linewidth = 2,\
            clip_on=False, marker='s', markersize=7 ,markeredgecolor=(254/255, 77/255, 79/255),markeredgewidth=2)

plt.plot(x, y_normalized, c=(114/255,170/255,207/255),label='Throughput',linewidth = 2,\
            clip_on=False, marker='^', markersize=7 ,markeredgecolor=(88/255,125/255,247/255),markeredgewidth=2)

#add x text
ax.set_title('Effect of Batch Size on Cost, Throughput, and Latency',fontsize='20')
ax.set_xlabel('Batch Size',fontsize='20')
ax.set_ylabel('Normalized Value',fontsize='20')
ax.legend(fontsize='18')
ax.grid(True)
# 修改横坐标的刻度
xticks(np.linspace(1,16,16,endpoint=True))
yticks(np.linspace(0,1,11,endpoint=True))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('figures/batch_size_tempo.pdf',format='pdf',bbox_inches='tight')
plt.show()
