import numpy as np
import cv2,os,matplotlib
import matplotlib.pyplot as plt
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

def count_width_height(image_path : str,label_path : str):
    #读取 labels
    width_pool= []
    height_pool = []
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    total_area = h * w
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    for x in lb:
        # 反归一化并得到左上和右下坐标，画出矩形框
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = xywh2xyxy(x, w, h)
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y    
        width_pool.append(width)
        height_pool.append(height)
    return width_pool,height_pool


if __name__ == '__main__':
    labels_path = '/Volumes/Livion/Pandadataset/标记/4k/'
    images_path = '/Volumes/Livion/Pandadataset/图片/4k/'
    labels = os.listdir(labels_path)
    images = os.listdir(images_path)
    width_pool = {'scene_01':[],'scene_02':[],'scene_03':[],'scene_04':[],'scene_05':[],'scene_06':[],'scene_07':[],'scene_08':[],'scene_09':[],'scene_10':[]}
    height_pool = {'scene_01':[],'scene_02':[],'scene_03':[],'scene_04':[],'scene_05':[],'scene_06':[],'scene_07':[],'scene_08':[],'scene_09':[],'scene_10':[]}
    for index,label in enumerate(labels):
        if label == 'SEQ_02_001.txt':
            break
        if label == '.DS_Store':
            continue
        label_path = os.path.join(labels_path, label)
        image_path = os.path.join(images_path, label.replace('txt', 'jpg'))
        width,height = count_width_height(image_path,label_path)    
        width_pool['scene_'+label.split('_')[1]].append(width)
        height_pool['scene_'+label.split('_')[1]].append(height)

    scene_01_height = []
    scene_01_width = []
    for i in range(len(width_pool['scene_01'])):
        scene_01_height.extend(height_pool['scene_01'][i])
        scene_01_width.extend(width_pool['scene_01'][i])

    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, marker='.',s=2)
        ax.set_xlim(0,300)
        ax.set_ylim(0,400)
        ax.set_xticks([0,50,100,150,200,250])
        ax.set_xlabel('RoI Width (pixel)',fontsize=20)
        ax.set_ylabel('RoI Height (pixel)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # now determine nice limits by hand:
        binwidth = 4
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins, orientation='vertical',density=True)
        ax_histx.tick_params(axis='both', which='major', labelsize=18)
        ax_histy.hist(y, bins=bins, orientation='horizontal',density=True)
        ax_histy.tick_params(axis='both', which='major', labelsize=18)
        

    fig = plt.figure(figsize=(8, 8))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    scatter_hist(scene_01_width, scene_01_height, ax, ax_histx, ax_histy)

    plt.savefig('figures/motivation3.pdf',format='pdf',bbox_inches='tight')
    plt.show()











