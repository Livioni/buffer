import sys,time,os,cv2
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from buffer import Table,Image

scene_name = 'partitions_01'
network_bandwidth = 80 # 10Mbps
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
switch = 0
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
table1 = Table(1024,1024,csv_record=True,logs=False)
time.sleep(1)
table2 = Table(1024,1024,csv_record=True,logs=False)
time.sleep(1)
table3 = Table(1024,1024,csv_record=True,logs=False)
start_time = time.perf_counter()
SLO = 0.7
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
        if switch == 0:
            if table1.push(image) == False:
                table2.push(image)
                switch = 1
        elif switch == 1:
            if table2.push(image) == False:
                table1.push(image)
                switch = 2
        elif switch == 2:
            if table3.push(image) == False:
                table1.push(image)
                switch = 0


end_time = time.perf_counter()
print('Total time: ',end_time-start_time)
time.sleep(5)
table1.show_info()
table2.show_info()
table3.show_info()


