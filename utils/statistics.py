import os,cv2,json
import os.path
import numpy as np
import matplotlib.pyplot as plt
  
def count_images_and_sizes(folder_path):  
    """  
    统计指定文件夹内所有图片的尺寸，并输出每种尺寸图片的数量  
    """  
    # 获取文件夹中所有文件的文件名和扩展名  
    files = os.listdir(folder_path)  
  
    # 统计每种尺寸图片的数量  
    sizes = []  
    height_pool = []
    width_pool = []
    for file in files:  
        file_path = os.path.join(folder_path, file)  
        if os.path.isfile(file_path):  
            file_size = os.path.getsize(file_path) 
            image_size = cv2.imread(file_path).shape 
            height_pool.append(image_size[0])
            width_pool.append(image_size[1])
            sizes.append((image_size, file))  
  
    # 输出每种尺寸图片的数量  
    for size, file in sizes:  
        print(f"{size} - {file}")  
    
    height_pool = np.array(height_pool)
    width_pool = np.array(width_pool)
    print("average height x width: ", height_pool.mean(), "x", width_pool.mean())
    print("min height x min width: ", height_pool.min(), "x", width_pool.min())
    print("max height x max width: ", height_pool.max(), "x", width_pool.max())
  

def load_json(file_path):
    tracks_file_path = os.path.join(file_path, "tracks.json")
    seqinfo_file_path = os.path.join(file_path, "seqinfo.json")
    with open(seqinfo_file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
        seq_num = content["seqLength"]
        f.close()
    track_id = []
    show_frame = {}
    with open(tracks_file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
        f.close()

    # for track in anno:
    #     for frame in track['frames']:
    #         if frame["frame id"] == frameid:
    #             pid = track['track id']
    #             xmin = int(frame["rect"]["tl"]["x"] * savewidth)
    #             ymin = int(frame["rect"]["tl"]["y"] * saveheight)
    #             xmax = int(frame["rect"]["br"]["x"] * savewidth)
    #             ymax = int(frame["rect"]["br"]["y"] * saveheight)
    #             box = convert(savewidth, saveheight, xmin, ymin, xmax, ymax)
    #             if '-' in ''.join(box) or float(box[0]) > 1.0 or float(box[1]) > 1.0 or float(box[2]) > 1.0 or float(box[3]) > 1.0:
    #                 print(saveheight, saveheight, xmin, ymin, xmax, ymax)
    #                 print(box)
    #                 break

    for i in content:
        track_id.append(i["track id"])
        show_frame_id = []
        for j in i["frames"]:
            xmin = j["rect"]["tl"]["x"]
            ymin = j["rect"]["tl"]["y"]
            xmax = j["rect"]["br"]["x"]
            ymax = j["rect"]["br"]["y"]
            if (min(xmin, xmax) < 0) or (min(ymin, ymax) < 0) or (max(xmin, xmax) > 1) or (max(ymin, ymax) > 1):
                continue
            show_frame_id.append(j["frame id"])
        show_frame[i["track id"]] = show_frame_id

    object_count = {}
    for i in range(1,seq_num+1):
        object_count[i] = 0
        for key, value in show_frame.items():
            if i in value:
                object_count[i] += 1

    print("average object count: ", sum(object_count.values())/len(object_count))
    t = np.array(list(object_count.keys()))
    num = np.array(list(object_count.values()))
    return t, num

def draw_curve(t, num, file_name='default.png',title='Ground Truth Object Num'):
    base_path = 'utils/video_info/object_count/'
    file_name = os.path.join(base_path, file_name)
    fig, ax = plt.subplots()
    ax.plot(t, num)
    ax.set(xlabel='Frame', ylabel='Object Num',title=title)
    ax.grid()
    fig.savefig(file_name, dpi=300)


def count_file_num(folder_path):
    FileNum = {}
    # os.listdir(filePath)会读取出当前文件夹下的文件夹和文件
    for file in os.listdir(folder_path): 
        if file == ".DS_Store":
            continue
        concat_path = os.path.join(folder_path, file)
        if os.path.isdir(concat_path): # 判断是否为文件夹
            FileNum[file] = len(os.listdir(concat_path)) # 统计文件夹下的文件夹和文件的总数
    # print(FileNum)
    frame = np.array(list(FileNum.keys()))
    t = np.linspace(1, len(frame), len(frame))
    num = np.array(list(FileNum.values()))
    return t, num



if __name__ == "__main__":
    # 每帧-目标数量 画图
    # folder_path = "utils/video_info/train_annos"
    # files = os.listdir(folder_path)  
    # for file in files:
    #     if file == ".DS_Store":
    #         continue
    #     file_path = os.path.join(folder_path, file)
    #     t, num = load_json(file_path)
    #     draw_curve(t, num, file_name=file+'.png', title=file)


    # 指定要统计的文件夹路径  
    folder_path = "/Users/livion/Documents/test_videos/partitions"
    t, num = count_file_num(folder_path)
    draw_curve(t, num, file_name='01_University_Canteen_partitions.png', title='Partition Num')


