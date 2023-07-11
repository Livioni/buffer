import numpy as np
import cv2,os,matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as lines
plt.style.use("seaborn-v0_8-darkgrid")
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"


#坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywh2xyxy(x, w1, h1):
    labels = ['person']  
    label, x, y, w, h = x
    # print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
    #边界框反归一化
    x_t = x*w1
    y_t = y*h1 
    h_t = h*h1
    w_t = w*w1

    #计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    # 绘图  rectangle()函数需要坐标为整数
    # cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

def compute_mask_area(mask: np.ndarray) -> int:
    total_area = 0
    for object in mask: 
        area = np.sum(object!=0)
        total_area += area

    return total_area

def compute_backgroud_proportion(image_path : str,label_path : str) -> int:
    #读取 labels
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    # 读取图像文件
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for x in lb:
        # 反归一化并得到左上和右下坐标，画出矩形框
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = xywh2xyxy(x, w, h)
        mask[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)] = img[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
    cv2.imwrite('/Volumes/Livion/motivation/{}'.format(image_path.split('/')[-1]), mask)
    return 


if __name__ == '__main__':
    labels_path = '/Volumes/Livion/Pandadataset/标记/4k/'
    images_path = '/Volumes/Livion/Pandadataset/图片/4k/'
    labels = os.listdir(labels_path)
    images = os.listdir(images_path)
    for index,label in enumerate(labels):
        if label == '.DS_Store':
            continue
        label_path = os.path.join(labels_path, label)
        image_path = os.path.join(images_path, label.replace('txt', 'jpg'))
        compute_backgroud_proportion(image_path, label_path)





