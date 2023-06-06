import sys,time,os,cv2,pandas as pd
import numpy as np
from datetime import datetime
from tools import read_response
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from utils.invoker import invoke_yolo_batch_v2
from cost_function import Ali_function_cost,Ali_idle_cost


scene_name = 'scene_01_part'
network_bandwidth = 10 # 10Mbps
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
## define file name and source file path
record_file_name = 'AmpereA10_vanilla_' + scene_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") 
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/vanilla-partitions/'
## prepare data
fields = ['Timestamp', 'File Name', 'File Size(Byte)', 'Transmission Time (ms)', 'Prepocess Time(ms)', 'Inference Time(ms)', \
          'Latency (ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)','Bandwidth(Mbps)']
data_frame = pd.DataFrame(columns=fields)
csv_file_path = save_csv_file_path + record_file_name +'.csv'
data_frame.to_csv(csv_file_path, index=False)
## trigger and infer 
files = os.listdir(source_file_path)
files.sort()
total_cost = 0
for file in files:
    file_path = source_file_path + file
    file_size = os.path.getsize(file_path)
    # transmission_time = file_size/upload_byte_per_second
    image = cv2.imread(file_path)
    image_numpy = np.array(image)
    new_image = np.expand_dims(image_numpy, axis=0)
    start_time = time.time()
    response,real_time = invoke_yolo_batch_v2(new_image)
    service_time, inference_time, prepocess_time = read_response(response)
    transmission_time = real_time - service_time
    idle_cost = Ali_idle_cost(Time=transmission_time,Mem=4,GPU=6)
    cost = Ali_function_cost(Time=service_time, Mem=4, CPU=2, GPU=6)
    total_cost += cost + idle_cost
    latency = transmission_time + service_time
    data_frame = pd.DataFrame([[start_time, file, file_size, round(transmission_time*1000,5), round(prepocess_time*1000,5),\
                                round(inference_time*1000,5),round(latency*1000,5), idle_cost, cost, network_bandwidth]], columns=fields)
    data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
    print('File: ', file, 'Inference Time: ', inference_time, 'Transmission Time:', transmission_time, 'Total time', latency, 'Cost: ', cost)
    # time.sleep(0.8)
print('Total Cost: ', total_cost)



