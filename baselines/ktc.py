import sys,time,os,cv2,pandas as pd
import numpy as np
from datetime import datetime
from tools import read_response
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from utils.invoker import invoke_yolo_batch_v1
from cost_function import Ali_function_cost, Ali_idle_cost
from buffer import Queue,Image

scene_name = 'partitions_08'
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
    scene_partition_name = 'scene_'+ scene_name[-2:] +'_part'
    record_file_name = 'ktc_' + scene_name + '_' + str(i)
    source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
    save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/ktc/' + 'scene_'+ scene_name[-2:] +'/'
    compare_source_path = '/Users/livion/Documents/test_videos/' + scene_partition_name + '/'
    if os.path.exists(save_csv_file_path) == False:
        os.makedirs(save_csv_file_path)
    ## prepare data
    fields = ['Timestamp', 'File Name', 'File Size(Byte)','Canvas Size', 'Batch Size', 'Image Number', 'Canvas Efficiency', 'Bin Packing Time(ms)', \
            'Prepocess Time(ms)','Inference Time (ms)', 'Transmission Time (ms)', 'Latency (ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)','Bandwidth(Mbps)']
    data_frame = pd.DataFrame(columns=fields)
    csv_file_path = save_csv_file_path + record_file_name +'.csv'
    data_frame.to_csv(csv_file_path, index=False)
    ## trigger and infer
    files = os.listdir(source_file_path)
    files.sort()
    minibatch = {}
    for file in files:
        if file == '.DS_Store':
            continue
        prefix = file.split('_')[0]
        if int(prefix) not in minibatch.keys():
            minibatch[int(prefix)] = []
        minibatch[int(prefix)].append(file)

    compare_files = os.listdir(compare_source_path)
    compare_files.sort()
    compare_file_size = {}
    for index, file in enumerate(compare_files):
        if file == '.DS_Store':
            continue
        file_path = compare_source_path + file
        file_size = os.path.getsize(file_path)
        compare_file_size[101+index] = file_size

    queue = Queue(100,1024,1024)
    canva_size = str(queue.width) + 'x' + str(queue.height)
    total_cost = 0
    for key,value in minibatch.items():
        total_file_size = 0
        image_path_list = [source_file_path + file for file in value]
        for img in image_path_list:
            image = Image(cv2.imread(img),time.time())
            file_size = os.path.getsize(img)
            total_file_size += file_size
            # if total_file_size > compare_file_size[key]:
            #     total_file_size = compare_file_size[key]
            queue.add(image)
        file = str(key) + '.jpg'
        start_time = time.perf_counter()
        canvas = queue.greedy_packer_solve()
        end_time = time.perf_counter()
        bin_pack_time = end_time - start_time
        canvas_efficency = queue.efficiency
        # transmission_time = total_file_size/upload_byte_per_second
        response,real_time = invoke_yolo_batch_v1(canvas)
        service_time, inference_time, prepocess_time = read_response(response)
        transmission_time = real_time - service_time
        idle_cost = Ali_idle_cost(Time=transmission_time+bin_pack_time, Mem=4, GPU=6)
        cost = Ali_function_cost(Time=service_time, Mem=4, CPU=2, GPU=6)
        total_cost += cost + idle_cost
        latency = transmission_time + service_time
    #fields = ['Timestamp', 'File Name', 'File Size(Byte)','Canvas Size', 'Batch Size', 'Image Number', 'Canvas Efficiency', 'Bin Packing Time(ms)'\
    #          'Prepocess Time(ms)','Inference Time (ms)', 'Transmission Time (ms)', 'Latency (ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)']
        data_frame = pd.DataFrame([[start_time, file, total_file_size,canva_size, len(canvas),len(image_path_list),\
                                    round(canvas_efficency,5), round(bin_pack_time*1000,5), round(prepocess_time*1000,5), \
                                    round(inference_time*1000,5),round(transmission_time*1000,5),round(latency*1000,5),\
                                    idle_cost, cost, network_bandwidth]], columns=fields)
        data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
        print('File: ', file, 'Inference Time: ', round(service_time,5),'Transmission Time:',transmission_time,'Latency:',latency, 'Cost', cost)
        queue.clear()

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
    avg_inference_time[i-1] = np.mean(data_frame['Inference Time (ms)'].values)
    inference_time_std[i-1] = np.std(data_frame['Inference Time (ms)'].values)
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
excel_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/ktc/' + 'scene_' + scene_name[-2:] + '/scene_' + scene_name[-2:] + '.xlsx'
df.to_excel(excel_file_path, index=False)







