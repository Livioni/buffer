import sys,time,os,cv2,pandas as pd
from datetime import datetime
from tools import read_response
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from utils.invoker import invoke_yolo_batch_v1
from cost_function import Ali_function_cost, Ali_idle_cost
from buffer import Queue,Image

scene_name = 'partitions_01'
network_bandwidth = 10 # 10Mbps
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
## define file name and source file path
scene_partition_name = 'scene_'+ scene_name[-2:] +'_part'
record_file_name = 'ktc_' + scene_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") 
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/ktc/'
compare_source_path = '/Users/livion/Documents/test_videos/' + scene_partition_name + '/'
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
    transmission_time = total_file_size/upload_byte_per_second
    response,_ = invoke_yolo_batch_v1(canvas)
    service_time, inference_time, prepocess_time = read_response(response)
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
    time.sleep(0.8)

print('Total Cost: ', total_cost)


