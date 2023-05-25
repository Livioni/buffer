import sys,time,os,pandas as pd
from datetime import datetime
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from utils.invoker import invoke_yolo_single
from cost_function import Ali_function_cost,Ali_idle_cost


scene_name = 'scene_01_part'
network_bandwidth = 1000
## define file name and source file path
record_file_name = 'vanilla_' + scene_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") 
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/vanilla-partitions/'
## prepare data
fields = ['Timestamp', 'File Name', 'File Size(Byte)', 'transmission Time (ms)', 'Latency(ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)']
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
    transmission_time = file_size/(network_bandwidth * 1000)
    start_time = time.time()
    time_taken = invoke_yolo_single(file_path)
    idle_cost = Ali_idle_cost(Time=transmission_time,Mem=4)
    cost = Ali_function_cost(Time=time_taken, Mem=4, CPU=2, GPU=4)
    total_cost += cost + idle_cost
    data_frame = pd.DataFrame([[start_time, file, file_size, round(transmission_time*1000,5), round(time_taken*1000,5),idle_cost, cost]], columns=fields)
    data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
    print('File: ', file, 'Time taken: ', time_taken, 'Cost: ', cost)
    time.sleep(1)
print('Total Cost: ', total_cost)



