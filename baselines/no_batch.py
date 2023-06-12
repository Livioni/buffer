import sys,time,os,cv2,pandas as pd
from datetime import datetime
import numpy as np
from tools import read_response
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from utils.invoker import invoke_yolo_single
from cost_function import Ali_function_cost, Ali_idle_cost

scene_name = 'partitions_01'
network_bandwidth = 10 # 10Mbps
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
## define file name and source file path
scene_partition_name = 'scene_'+ scene_name[-2:] +'_part'
record_file_name = 'A10No_batch_' + scene_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") 
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/no_batch/'
compare_source_path = '/Users/livion/Documents/test_videos/' + scene_partition_name + '/'
## prepare data
fields = ['Timestamp', 'File Name', 'File Size(Byte)', 'Prepocess Time(ms)','Inference Time (ms)', \
          'Transmission Time (ms)', 'Latency (ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)', 'Bandwidth(Mbps)']
data_frame = pd.DataFrame(columns=fields)
csv_file_path = save_csv_file_path + record_file_name +'.csv'
data_frame.to_csv(csv_file_path, index=False)
## trigger and infer
files = os.listdir(source_file_path)
files.sort()

total_cost = 0
for index, file in enumerate(files):
    if file == '.DS_Store':
        continue
    file_path = source_file_path + file
    file_size = os.path.getsize(file_path)
    image = cv2.imread(file_path)
    image_numpy = np.array(image)
    image_numpy = np.expand_dims(image_numpy, axis=0)
    transmission_time = file_size/upload_byte_per_second
    response,_ = invoke_yolo_single(image_numpy)
    service_time, inference_time, prepocess_time = read_response(response)
    idle_cost = Ali_idle_cost(Time=transmission_time, Mem=4, GPU=6)
    cost = Ali_function_cost(Time=service_time, Mem=4, CPU=2, GPU=6)
    total_cost += cost + idle_cost
    latency = transmission_time + service_time
#fields = ['Timestamp', 'File Name', 'File Size(Byte)', 'Prepocess Time(ms)','Inference Time (ms)', \
#          'Transmission Time (ms)', 'Latency (ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)']
    data_frame = pd.DataFrame([[index, file, file_size, round(prepocess_time*1000,5), round(inference_time*1000,5), \
                                round(transmission_time*1000,5),round(latency*1000,5),idle_cost, cost, network_bandwidth]], columns=fields)
    data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
    print('File: ', file, 'Inference Time: ', round(service_time,5),'Transmission Time:',transmission_time,'Latency:',latency, 'Cost', cost)

print('Total Cost: ', total_cost)


