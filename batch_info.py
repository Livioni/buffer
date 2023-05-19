import cv2,os,json,time
from buffer import Image, Queue
from utils.invoker import invoke_yolo,invoke_yolo_batch_v1

if __name__ == "__main__":
    # 创建一个2000x2000的画布，图片容量100
    box = Queue(100, 2000, 2000)
    # 读取图片
    files_path = '/Users/livion/Documents/2x2/partitions_01'
    files = os.listdir(files_path)
    files.sort()
    for index, file in enumerate(files):
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(files_path, file))
            created_time = time.time()  
            box.add(Image(img,created_time))
        if index == 5:
            break
    # 拼图
    result = box.greedy_packer_solve(visualize=False)
    print('Box efficieny:',box.efficiency)
    Qos_pool = []
    inference_time_pool = []
    for i in range(100):
        #调用yolo
        response,time_taken = invoke_yolo_batch_v1(result)
        json_result = json.loads(response)
        print('Inference_time:',json_result['Inference_time'])
        Qos_pool.append(time_taken)
        inference_time_pool.append(json_result['Inference_time'])
    average_Qos = sum(Qos_pool)/len(Qos_pool)
    average_inference_time = sum(inference_time_pool)/len(inference_time_pool)
    print('Average Qos:',average_Qos)
    print("Max QoS:",max(Qos_pool))
    print("Sorted QoS",sorted(Qos_pool))
    print('Average inference time:',average_inference_time)

