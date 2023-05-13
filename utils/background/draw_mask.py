import cv2 as cv
import numpy as np

src = cv.imread("/Volumes/Livion/Pandadataset/panda/images/scene_01_full/SEQ_01_101.jpg")
cv.imshow("input", src)
h, w, c = src.shape

# 手工绘制ROI区域
mask = np.zeros((h, w), dtype=np.uint8)
x_data = np.array([0,3840,3840,0])
y_data = np.array([660, 660, 2160, 2160])
pts = np.vstack((x_data, y_data)).astype(np.int32).T
cv.fillPoly(mask, [pts], (255), 8, 0)
cv.imshow("mask", mask)

# 根据mask，提取ROI区域
result = cv.bitwise_and(src, src, mask=mask)
cv.imshow("result", result)
cv.waitKey(0)