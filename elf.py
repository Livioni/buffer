import sys,time,os,cv2,pandas,threading
from utils.invoker import invoke_yolo_single
from buffer import Image
from baselines.tools import read_response
from baselines.cost_function import Ali_function_cost_usd
from threading import Thread


scene_name = 'partitions_01'
network_bandwidth = 80 # 10Mbps6
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
# define file name and source file path
source_file_path = '/Users/livion/Documents/4x4不带切割/' + scene_name + '/'
## prepare the source file list
files = os.listdir(source_file_path)
files.sort()
file_per_frame = {}
file_per_frame_size = {}
for file in files:
    if file == '.DS_Store':
        continue
    if int(file[:3]) not in file_per_frame.keys():
        file_per_frame[int(file[:3])] = [file]
        file_per_frame_size[int(file[:3])] = os.path.getsize(source_file_path + file)
    else:
        file_per_frame[int(file[:3])].append(file)
        file_per_frame_size[int(file[:3])] += os.path.getsize(source_file_path + file)


SLO = 1
for i in range(1,4):
    record_file_name = str(i) + 'elf_' + str(SLO) + '_bandwidth' + str(network_bandwidth) + '_' + scene_name
    fields = ['Timestamp', 'SLO', 'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Cost(USD)']
    data_frame = pandas.DataFrame(columns=fields)
    csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/logs/csv/'+record_file_name+'.csv'
    data_frame.to_csv(csv_file_path, index=False)
    csv_data = []

    def post_patch(image: Image,csv_data_local):
        response = invoke_yolo_single(image.image)
        service_time, inference_time, prepocess_time = read_response(response)
        whether_violated = 'No' if time.time() > ddl else 'Yes'
        cost = Ali_function_cost_usd(service_time,Mem=4,CPU=2,GPU=6)
        logs = [patch_cnt,whether_violated,prepocess_time,inference_time,service_time,cost]
        print("Whether violated: ",whether_violated,"Latency:", service_time, "Cost: ",cost)
        mutex.acquire()
        csv_data_local.append(logs)
        mutex.release()

    # SLO-aware algorithm
    patch_cnt = 0
    mutex = threading.Lock()
    start_time = time.perf_counter()
    for index, files in file_per_frame.items():
        ddl = time.time() + SLO      
        for image in files:
            if image == '.DS_Store':
                continue
            patch_cnt += 1
            img = cv2.imread(source_file_path + image)
            file_size = os.path.getsize(source_file_path + image)
            delay_time = file_size / upload_byte_per_second   
            image = Image(img,time.time(),ddl)
            time.sleep(delay_time)
            t = Thread(target=post_patch, args=(image,csv_data,))
            t.start()
        time.sleep(1)

    csv_data.sort(key=lambda x:x[0])
    for logs in csv_data:
        data_frame.loc[len(data_frame)]= logs 
        data_frame.to_csv(csv_file_path, index=False)

    end_time = time.perf_counter()
    print('Total time: ',end_time-start_time)




