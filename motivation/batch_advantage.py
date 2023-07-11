import sys,cv2,time,matplotlib,pandas as pd
import numpy as np
from datetime import datetime
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer') 
from baselines.cost_function import Ali_function_cost
from baselines.tools import read_response
from utils.invoker import invoke_yolo_single, invoke_yolo_batch_v3
from pylab import xticks,yticks


save_csv_file_path = 'motivation/batch_size_csv/'
fields = ['Index', 'Service Time (ms)', 'Inference Time (ms)', 'Prepocess Time (ms)', 'Cost ','Inference Time Per Frame','Service Time Per-frame (ms)', 'Cost Per-Frame']


image_path = 'motivation/test.jpg'
image = cv2.imread(image_path)
image_numpy = np.array(image)
image_numpy = np.expand_dims(image_numpy, axis=0)

# batch
# image_batch = np.repeat(image_numpy, 2, axis=0)
# response,time_taken = invoke_yolo_batch_v3(image_batch)


y = np.zeros(16)
yerr = np.zeros(16) 
cost = np.zeros(16)
cost_std = np.zeros(16)
qos = np.zeros(16)
qos_std = np.zeros(16)

# single
for ind,batch_size in enumerate(np.arange(1,17)):
    record_file_name = str(batch_size) + 'Motivation' + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") 
    data_frame = pd.DataFrame(columns=fields)
    csv_file_path = save_csv_file_path + record_file_name +'.csv'
    data_frame.to_csv(csv_file_path, index=False)
    for i in range(1,51):
        image_batch = np.repeat(image_numpy, batch_size, axis=0)
        response,time_taken = invoke_yolo_batch_v3(image_batch)
        service_time, inference_time, prepocess_time = read_response(response)
        cost_1 = Ali_function_cost(Time=service_time, Mem=4, CPU=2, GPU=8,basic_cost=0)
        service_time = round(service_time*1000,5)
        inference_time = round(inference_time*1000,5)
        prepocess_time = round(prepocess_time*1000,5)
        service_time_per_frame = round(service_time/batch_size,5)
        inference_time_per_frame = round(inference_time/batch_size,5)
        cost_per_frame = round(cost_1/batch_size,10)
        data_frame = pd.DataFrame([[i, service_time, inference_time, prepocess_time, cost_1,inference_time_per_frame, service_time_per_frame, cost_per_frame]], columns=fields)
        data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
        print("Index: {}, service_time: {}, inference_time: {}, prepocess_time: {}".format(i,service_time,inference_time,prepocess_time))
        # time.sleep(0.5)
    # 读取csv文件
    df = pd.read_csv(csv_file_path)
    # 读取某一列
    latency = df['Service Time (ms)'].values
    latency_per_frame = df['Inference Time Per Frame'].values
    cost_per_frame = df['Cost Per-Frame'].values

    latency_per_frame_avg = np.mean(latency_per_frame)
    latency_per_frame_std = np.std(latency_per_frame)
    cost_per_frame_avg = np.mean(cost_per_frame)
    cost_per_frame_std = np.std(cost_per_frame)
    latency_avg = np.mean(latency)  
    latency_std = np.std(latency)

    df['latency_per_frame_avg'] = latency_per_frame_avg
    df['latency_per_frame_std'] = latency_per_frame_std
    df['cost_per_frame_avg'] = cost_per_frame_avg
    df['cost_per_frame_std'] = cost_per_frame_std
    df['latency_avg'] = latency_avg
    df['latency_std'] = latency_std

    # 将DataFrame写回CSV文件
    df.to_csv(csv_file_path, index=False)

    print("latency_per_frame_avg: {}, latency_per_frame_std: {}".format(latency_per_frame_avg,latency_per_frame_std))
    print("cost_per_frame_avg: {}, cost_per_frame_std: {}".format(cost_per_frame_avg,cost_per_frame_std))

    y[ind] = latency_per_frame_avg
    yerr[ind] = latency_per_frame_std
    cost[ind] = cost_per_frame_avg
    cost_std[ind] = cost_per_frame_std
    qos[ind] = latency_avg
    qos_std[ind] = latency_std

np.save('motivation/npy/y.npy',y)
np.save('motivation/npy/yerr.npy',yerr)
np.save('motivation/npy/cost.npy',cost)
np.save('motivation/npy/cost_std.npy',cost_std)
np.save('motivation/npy/qos.npy',qos)
np.save('motivation/npy/qos_std.npy',qos_std)

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

x = np.linspace(1,16,16)

def min_max_normalization(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

cost_normalized = min_max_normalization(cost)
cost_normalized_std = cost_std/(np.max(cost)-np.min(cost))

y_normalized = min_max_normalization(1/y)
yerr_normalized = yerr/(np.max(y)-np.min(y))

qos_normalized = min_max_normalization(qos)
qos_normalized_std = qos_std/(np.max(qos)-np.min(qos))

fig, ax = plt.subplots()
plt.plot(x, cost_normalized,c=(253/255,185/255,107/255),label='Cost Per-Frame',linewidth = 2, \
            clip_on=False, marker='o', markersize=7 ,markeredgecolor=(249/255, 172/255, 19/255),markeredgewidth=2)

plt.plot(x, qos_normalized,c=(254/255, 162/255, 158/255),label='Latency',linewidth = 2,\
            clip_on=False, marker='s', markersize=7 ,markeredgecolor=(254/255, 77/255, 79/255),markeredgewidth=2)

plt.plot(x, y_normalized, c=(114/255,170/255,207/255),label='Throughput',linewidth = 2,\
            clip_on=False, marker='^', markersize=7 ,markeredgecolor=(88/255,125/255,247/255),markeredgewidth=2)

#add x text
ax.set_title('Effect of Batch Size on Cost, Throughput, and Latency',fontsize='14')
ax.set_xlabel('Batch Size',fontsize='14')
ax.set_ylabel('Normalized Value',fontsize='14')
ax.legend(fontsize='12')
ax.grid(True)
# 修改横坐标的刻度
xticks(np.linspace(1,16,16,endpoint=True))
yticks(np.linspace(0,1,11,endpoint=True))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('figures/batch_size_111.pdf',format='pdf',bbox_inches='tight')
plt.show()
