import cv2,os,json
from buffer import Image, Queue
from utils.invoker import invoke_yolo

if __name__ == "__main__":
    # 创建一个2000x2000的画布，图片容量100
    box = Queue(100, 2000, 2000)
    # 读取图片
    files_path = '/Users/livion/Documents/test_videos/partitions_01'
    files = os.listdir(files_path)
    for index, file in enumerate(files):
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(files_path, file))
            box.add(Image(img))
        if index == 3:
            break
    # 拼图
    result = box.greedy_packer_solve(visualize=False)
    #调用yolo
    response = invoke_yolo(result)
    json_result = json.loads(response)
    print(json_result['Inference_time'])

