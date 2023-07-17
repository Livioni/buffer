import sys,time,os,cv2
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from buffer import Fixed_Table,Image

scene_name = 'partitions_01'
network_bandwidth = 80 # 10Mbps6
upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
## define file name and source file path
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

# SLO-aware algorithm
batch_size = 10
SLO = 1
record_file_name1 = '3Batch=' + str(batch_size)+'_SLO=' + str(SLO) + 's_1'
record_file_name2 = '3Batch=' + str(batch_size)+'_SLO=' + str(SLO) + 's_2'
table1 = Fixed_Table(record_file_name=record_file_name1,batch_size=batch_size, csv_record=True)
table2 = Fixed_Table(record_file_name=record_file_name2,batch_size=batch_size, csv_record=True)
start_time = time.perf_counter()
batch_cnt = 0
switch = False
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
        if switch == False:
            table1.push(image)
        else: 
            table2.push(image)
        batch_cnt += 1
        if batch_cnt == 8:
            switch = ~switch
            batch_cnt = 0

table1.trigger()
table2.trigger()
print(table1.inference_round)
print(table2.inference_round)
end_time = time.perf_counter()
print('Total time: ',end_time-start_time)
time.sleep(3)
import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('logs/csv/'+record_file_name1+'.csv')
df2 = pd.read_csv('logs/csv/'+record_file_name2+'.csv')

# 合并两个数据帧
df = pd.concat([df1, df2])

# 按第一列排序，假设第一列的名称为 'column1'
# df = df.sort_values(by='Timestamp')

# 保存到新的CSV文件
df.to_csv('logs/Batch=' + str(batch_size)+ '_SLO=' + str(SLO) + 's_1.csv', index=True)



