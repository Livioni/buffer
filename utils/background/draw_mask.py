import cv2 as cv
import numpy as np
import yaml

src = cv.imread("/Volumes/Livion/Pandadataset/panda/images/scene_03_full/SEQ_03_101.jpg")
cv.imshow("input", src)
h, w, c = src.shape
configration_path = "utils/background/configuration.yaml"

def read_yaml_all(yaml_path):
    try:
        # 打开文件
        with open(yaml_path,"r",encoding="utf-8") as f:
            data=yaml.load(f,Loader=yaml.FullLoader)
            return data
    except:
        return None
    
configration = read_yaml_all(configration_path)


# 手工绘制ROI区域
mask = np.zeros((h, w), dtype=np.uint8)
for num,value in configration.items():
    x_data = np.array(value['x_data'])
    y_data = np.array(value['y_data'])
pts = np.vstack((x_data, y_data)).astype(np.int32).T
cv.fillPoly(mask, [pts], (255), 8, 0)
cv.imshow("mask", mask)

# 根据mask，提取ROI区域
result = cv.bitwise_and(src, src, mask=mask)
cv.imshow("result", result)
cv.waitKey(0)