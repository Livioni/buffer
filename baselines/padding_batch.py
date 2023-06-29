import sys,time,os,cv2
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from buffer import Fixed_Table,Image

scene_name = 'partitions_01'
network_bandwidth = 80 # 10Mbps
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
## define file name and source file path
source_file_path = '/Users/livion/Documents/4x4(带切割）/' + scene_name + '/'
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

# SLO-aware algorithm
batch_size = 10
SLO = 0.6
record_file_name = '1Batch=' + str(batch_size)+'_SLO=' + str(SLO) + 's.csv'
table1 = Fixed_Table(record_file_name=record_file_name,batch_size=batch_size, csv_record=True)
start_time = time.perf_counter()
for index, files in file_per_frame.items():
    ddl = time.time() + SLO      
    for image in files:
        if image == '.DS_Store':
            continue
        img = cv2.imread(source_file_path + image)
        file_size = os.path.getsize(source_file_path + image)
        delay_time = file_size / upload_byte_per_second   
        image = Image(img,time.time(),ddl)
        time.sleep(delay_time)
        table1.push(image)

table1.trigger()
print(table1.inference_round)
end_time = time.perf_counter()
print('Total time: ',end_time-start_time)
time.sleep(3)


