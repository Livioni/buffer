import sys,time,os,cv2
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from buffer import Table,Image

scene_name = 'partitions_01'
network_bandwidth = 10000 #bytes/s
switch = False
## define file name and source file path
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
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
table1 = Table(1024,1024,0.12)
time.sleep(1)
table2 = Table(1024,1024,0.12)
start_time = time.perf_counter()

for index, files in file_per_frame.items():
    SLO = 0.55
    for image in files:
        if image == '.DS_Store':
            continue
        img = cv2.imread(source_file_path + image)
        file_size = os.path.getsize(source_file_path + image)
        delay_time = file_size / (network_bandwidth * 1000)             
        image = Image(img,time.time(),SLO)
        # time.sleep(delay_time)
        # SLO -= delay_time
        if switch == False:
            if table1.push(image) == False:
                table2.push(image)
                switch = True
        else:
            if table2.push(image) == False:
                table1.push(image)
                switch = False
    wait_time = 0.1 - file_per_frame_size[index] / (network_bandwidth * 1000)
    print('wait time: ',wait_time)
    time.sleep(wait_time)

end_time = time.perf_counter()
time.sleep(5)
table1.show_info()
table2.show_info()
print('Total time: ',end_time-start_time)
