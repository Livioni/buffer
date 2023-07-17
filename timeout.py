import sys,time,os,cv2,pandas,threading
from buffer import Image,Timeout
from threading import Thread
import numpy as np


scene_name = 'partitions_01'
bandwidths = 80 # 10Mbps6
upload_byte_per_second = bandwidths * 1000 * 1000 / 8
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

SLO = 0.8
timeout_table = Timeout(record_file_name = 'timeout_' + str(SLO) + '_bandwidth' + str(bandwidths) + '_2',time_out=0.08)
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
        timeout_table.push(image)

end_time = time.perf_counter()
time.sleep(10)
print('Total time: ',end_time-start_time)
del timeout_table
 



