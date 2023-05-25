import sys,time,os,cv2,pandas as pd
from datetime import datetime
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from utils.invoker import invoke_yolo_batch_v1
from cost_function import Ali_function_cost, Ali_idle_cost
from buffer import Queue,Image

scene_name = 'partitions_01'
network_bandwidth = 1000
## define file name and source file path
record_file_name = 'ktc_' + scene_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") 
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
save_csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/data/ktc/'
## prepare data
fields = ['Timestamp', 'File Name', 'File Size(Byte)','Canvas Size', 'Batch Size', 'Image Number', 'Canvas Efficiency', 'transmission Time (ms)', 'Latency(ms)', 'Idle Cost (CNY)', 'Trigger Cost(CNY)']
data_frame = pd.DataFrame(columns=fields)
csv_file_path = save_csv_file_path + record_file_name +'.csv'
data_frame.to_csv(csv_file_path, index=False)
## trigger and infer
files = os.listdir(source_file_path)
files.sort()
minibatch = {}
for file in files:
    prefix = file.split('_')[0]
    if int(prefix) not in minibatch.keys():
        minibatch[int(prefix)] = []
    minibatch[int(prefix)].append(file)


queue = Queue(100,2016,2016)
canva_size = str(queue.width) + 'x' + str(queue.height)
total_cost = 0
for key,value in minibatch.items():
    total_file_size = 0
    image_path_list = [source_file_path + file for file in value]
    for img in image_path_list:
        image = Image(cv2.imread(img),time.time())
        file_size = os.path.getsize(img)
        total_file_size += file_size
        queue.add(image)
    file = str(key) + '.jpg'
    start_time = time.time()
    canvas = queue.greedy_packer_solve()
    end_time = time.time()
    bin_pack_time = end_time - start_time
    canvas_efficency = queue.efficiency
    transmission_time = total_file_size/(network_bandwidth * 1000)
    _,time_taken = invoke_yolo_batch_v1(canvas)
    idle_cost = Ali_idle_cost(Time=transmission_time+bin_pack_time, Mem=4)
    cost = Ali_function_cost(Time=time_taken, Mem=4, CPU=2, GPU=4)
    total_cost += cost + idle_cost
    data_frame = pd.DataFrame([[start_time, file, total_file_size,canva_size, len(canvas),\
                                len(image_path_list),round(canvas_efficency,5),round(transmission_time*1000,5),\
                                round(time_taken*1000,5),idle_cost, cost]], columns=fields)
    data_frame.to_csv(csv_file_path, index=False, mode='a', header=False)
    print('File: ', file, 'Time taken: ', round(time_taken,5),'Cost', cost)
    queue.clear()
    time.sleep(1)

print('Total Cost: ', total_cost)


