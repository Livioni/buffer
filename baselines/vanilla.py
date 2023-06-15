import sys,time,os,cv2,pandas as pd
import numpy as np
from datetime import datetime
from tools import read_response
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')        
from utils.invoker import invoke_yolo_batch_v2
from cost_function import Ali_function_cost, Ali_idle_cost


scene_name = 'scene_10_full'
network_bandwidth = 10 # 10Mbps
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
duplication = 10

#### Data to save
cost_data = np.zeros(duplication)
avg_file_size = np.zeros(duplication)
avg_transmission_time = np.zeros(duplication)
avg_preprocess_time = np.zeros(duplication)
avg_inference_time = np.zeros(duplication)
avg_latency = np.zeros(duplication)
avg_idle_cost = np.zeros(duplication)
avg_trigger_cost = np.zeros(duplication)  

file_size_std = np.zeros(duplication)
transmission_time_std = np.zeros(duplication)
preprocess_time_std = np.zeros(duplication)
inference_time_std = np.zeros(duplication)
latency_std = np.zeros(duplication)
idle_cost_std = np.zeros(duplication)
trigger_cost_std = np.zeros(duplication)

for i in range(1,duplication+1):
## define file name and source file path
    record_file_name = 'AmpereA10_vanilla_' + scene_name + '_' + str(i)
    source_file_path = '/Volumes/Livion/Pandadataset/panda/images/' + scene_name + '/'
    save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/vanilla/'+ scene_name[0:8] +'/'
    if os.path.exists(save_csv_file_path) == False:
        os.makedirs(save_csv_file_path)
    ## prepare data
    fields = ['Timestamp', 'File Name', 'File Size(Byte)', 'Transmission Time (ms)', 'Prepocess Time(ms)', 'Inference Time(ms)', 'Latency (ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)','Bandwidth(Mbps)']
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
        #version1: invoke_yolo_batch_v1
        response,real_time = invoke_yolo_batch_v2(new_image)
        service_time, inference_time, prepocess_time = read_response(response)
        transmission_time = real_time - service_time
        idle_cost = Ali_idle_cost(Time=transmission_time, Mem=4, GPU=6)
        cost = Ali_function_cost(Time=service_time, Mem=4, CPU=2, GPU=6)
        total_cost += cost + idle_cost
        latency = transmission_time + service_time
        data_frame = pd.DataFrame([[start_time, file, file_size, round(transmission_time*1000,5), round(prepocess_time*1000,5),\
                                    round(inference_time*1000,5),round(latency*1000,5), idle_cost, cost, network_bandwidth]], columns=fields)
        data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
        print('File: ', file, 'Inference Time: ', inference_time, 'Transmission Time:', transmission_time, 'Total time', latency, 'Cost: ', cost)
        # time.sleep(0.8)
    print('Total Cost: ', total_cost)
    # read csv file
    data_frame = pd.read_csv(csv_file_path)  
    cost_data[i-1] = total_cost
    avg_file_size[i-1] = np.mean(data_frame['File Size(Byte)'].values)
    file_size_std[i-1] = np.std(data_frame['File Size(Byte)'].values)
    avg_transmission_time[i-1] = np.mean(data_frame['Transmission Time (ms)'].values)
    transmission_time_std[i-1] = np.std(data_frame['Transmission Time (ms)'].values)
    avg_preprocess_time[i-1] = np.mean(data_frame['Prepocess Time(ms)'].values)
    preprocess_time_std[i-1] = np.std(data_frame['Prepocess Time(ms)'].values)
    avg_inference_time[i-1] = np.mean(data_frame['Inference Time(ms)'].values)
    inference_time_std[i-1] = np.std(data_frame['Inference Time(ms)'].values)
    avg_latency[i-1] = np.mean(data_frame['Latency (ms)'].values)
    latency_std[i-1] = np.std(data_frame['Latency (ms)'].values)
    avg_idle_cost[i-1] = np.mean(data_frame['Idle Cost (CNY)'].values)
    idle_cost_std[i-1] = np.std(data_frame['Idle Cost (CNY)'].values)
    avg_trigger_cost[i-1] = np.mean(data_frame['Trigger Cost(CNY)'].values)
    trigger_cost_std[i-1] = np.std(data_frame['Trigger Cost(CNY)'].values)
        
# 创建一个DataFrame
data_2_excel = {
    'Index' : np.arange(1,duplication+1),
    'Total Cost': cost_data,
    'Average File Size': avg_file_size,
    'File Size Std': file_size_std,
    'Average Transmission Time': avg_transmission_time,
    'Transmission Time Std': transmission_time_std,
    'Average Preprocess Time': avg_preprocess_time,
    'Preprocess Time Std': preprocess_time_std,
    'Average Inference Time': avg_inference_time,
    'Inference Time Std': inference_time_std,
    'Average Latency': avg_latency,
    'Latency Std': latency_std,
    'Average Idle Cost': avg_idle_cost,
    'Idle Cost Std': idle_cost_std,
    'Average Trigger Cost': avg_trigger_cost,
    'Trigger Cost Std': trigger_cost_std
}
df = pd.DataFrame(data_2_excel) 
excel_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/vanilla/'+ scene_name[0:8] + '/' + scene_name[0:8] + '.xlsx'
df.to_excel(excel_file_path, index=False)



