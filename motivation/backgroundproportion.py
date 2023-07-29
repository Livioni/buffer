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
    total_area = h * w
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in lb:
        # 反归一化并得到左上和右下坐标，画出矩形框
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = xywh2xyxy(x, w, h)
        mask[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)] = 255
    area = compute_mask_area(mask)
    proportion = area / total_area
    return proportion


if __name__ == '__main__':
    # labels_path = '/Volumes/Livion/Pandadataset/标记/4k/'
    # images_path = '/Volumes/Livion/Pandadataset/图片/4k/'
    # labels = os.listdir(labels_path)
    # images = os.listdir(images_path)
    # proportion_dict = {'scene_01':[],'scene_02':[],'scene_03':[],'scene_04':[],'scene_05':[],'scene_06':[],'scene_07':[],'scene_08':[],'scene_09':[],'scene_10':[]}
    # proportion_keys = list(proportion_dict.keys())
    # for index,label in enumerate(labels):
    #     if label == '.DS_Store':
    #         continue
    #     label_path = os.path.join(labels_path, label)
    #     image_path = os.path.join(images_path, label.replace('txt', 'jpg'))
    #     proportion = compute_backgroud_proportion(image_path, label_path)
    #     print("第{}张图片的ROI比例为：{}".format(image_path,proportion))
    #     proportion_dict['scene_'+label.split('_')[1]].append(proportion)

    # # for key,value in proportion_dict.items():
    # #     proportion_dict[key] = np.mean(value)

    # # print("平均ROI比例为：{}".format(proportion_dict))
    # np.save('/Users/livion/Documents/GitHub/Sources/buffer/utils/video_info/data/proportion.npy',proportion_dict)
    proportion_dict = np.load('/Users/livion/Documents/GitHub/Sources/buffer/utils/video_info/data/proportion.npy',allow_pickle=True).item()
    new_dict = {}
    for key,value in proportion_dict.items():
        new_dict[key] = np.mean(value)

    print("平均ROI比例为：{}".format(new_dict))
    labels = ['scene_01', 'scene_02', 'scene_03', 'scene_04', 'scene_05', 'scene_06', 'scene_07', 'scene_08', 'scene_09', 'scene_10']
    linestyles = ['solid','solid','solid','solid','solid','solid','solid','solid','solid','solid']
    # linestyles = ['-.','-.','-.','-.','-.','-.','-.','-.','-.','-.']
    fig, ax1 = plt.subplots(1, 1,figsize=(12,5))
    for index,area in enumerate(proportion_dict.values()):
        ax1.plot(range(1,len(area)+1), area, linestyle=linestyles[index], marker = '.',markersize='5.5', linewidth=2,label = labels[index])
        ax1.grid()
    
    ax1.legend(labels,ncols=5,bbox_to_anchor=(1.0, 1.3), loc='upper right',fontsize=18)
    ax1.set_ylabel('Region of Interest Proportion',fontsize='24')
    ax1.set_xlabel('Frame Index',fontsize='24')
    ax1.set_xlim(-1,240)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig('figures/motivation1.pdf',format='pdf',bbox_inches='tight')
    # plt.show()

    fig, ax2 = plt.subplots(1, 1,figsize=(12,4))
    prop = [np.mean(area) for area in proportion_dict.values()]
    for index,x in enumerate(proportion_dict.values()):
        sns.ecdfplot(data=x,ax=ax2,label=labels[index],linewidth=3)

    ax2.grid(True)
    ax2.set_ylabel('CDF',fontsize='24')
    ax2.set_xlabel('Region of Interest Proportion',fontsize='24')
    ax2.tick_params(axis='both', which='major', labelsize=22)

    plt.tight_layout()
    plt.savefig('figures/motivation2.pdf',format='pdf',bbox_inches='tight')
    plt.show()







