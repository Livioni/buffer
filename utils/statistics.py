import os,cv2
import os.path  
import numpy as np
  
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
  
# 指定要统计的文件夹路径  
folder_path = "/Users/livion/Documents/GitHub/Sources/buffer/images"  
  
# 调用函数统计图片尺寸和数量  
count_images_and_sizes(folder_path)