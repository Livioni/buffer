import sys,time,os,cv2
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from buffer import Table,Image

scene_name = 'partitions_01'
network_bandwidth = 1000
switch = False
## define file name and source file path
source_file_path = '/Users/livion/Documents/test_videos/' + scene_name + '/'
## prepare the source file list
files = os.listdir(source_file_path)
files.sort()

# SLO-aware algorithm
table1 = Table(1024,1024,0.165)
time.sleep(1)
table2 = Table(1024,1024,0.165)
start_time = time.time()
for image in files:
    img = cv2.imread(source_file_path + image)
    file_size = os.path.getsize(source_file_path + image)
    delay_time = file_size / (network_bandwidth * 1000)
    image = Image(img,time.time(),1.4)
    time.sleep(delay_time)
    if switch == False:
        if table1.push(image) == False:
            table2.push(image)
            switch = True
    else:
        if table2.push(image) == False:
            table1.push(image)
            switch = False
end_time = time.time()
time.sleep(5)
table1.show_info()
table2.show_info()
print('Total time: ',end_time-start_time)

