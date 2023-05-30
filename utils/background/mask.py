# _*_ coding: utf-8 _*_
# @name : GetROI.py
# @author : 霜晨月~
import cv2
import numpy as np
class GetRoiMouse():
    global img # 输入的图像

    def __init__(self):
        self.lsPointsChoose = [] 
        self.tpPointsChoose = [] 
        self.pointsCount = 0 	# 顶点计数
        self.pointsMax = 20		# 最大顶点个数
        self.mouseWindowName = 'click ROI'

    def mouseclick(self):	# 显示一个窗口
        cv2.namedWindow(self.mouseWindowName)
        # opecv可以设置监听鼠标
        # setMouseCallback(windowName,onMouse())
        # 在onMouse中写点击鼠标时要进行的工作
        cv2.setMouseCallback(self.mouseWindowName, self.on_mouse)
    
        cv2.imshow(self.mouseWindowName, img)
        cv2.waitKey(0)


    def getRoi(self):
        mask = np.zeros(img.shape, np.uint8)
        pts = np.array(self.lsPointsChoose, np.int32)  # 顶点集
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
        mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
        # 顶点个数：4，矩阵变成4*1*2维
        # OpenCV中需要将多边形的顶点坐标变成顶点数×1×2维的矩阵
        # 这里 reshape 的第一个参数为-1, 表示“任意”，意思是这一维的值是根据后面的维度的计算出来的
        # void fillPoly(InputOutputArray img, InputArrayOfArrays pts,
        #               const Scalar& color, int lineType=8, int shift=0, Point offset=Point() )
        # cv2.imshow('mask', mask2)
        # cv2.imwrite('mask.bmp', mask2)
        # cv2.drawContours(mask,points,-1,(255,255,255),-1)
        # 截取图像中的对应位置
        self.ROI = cv2.bitwise_and(mask2, img)
	
	# OpenCV的鼠标响应函数，可以在内部定义鼠标的各种响应
    def on_mouse(self, event, x, y, flags, param):
        # 左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            print('left-mouse')
            self.pointsCount += 1
            print(self.pointsCount)
            point1 = (x, y)
            print(x, y)
            # 画出点击的位置
            img1=img.copy()
            cv2.circle(img1, point1, 10, (0, 255, 0), 2)
            self.lsPointsChoose.append([x, y])
            self.tpPointsChoose.append((x, y))
            # 将鼠标选的点用直线连起来
            for i in range(len(self.tpPointsChoose) - 1):
                cv2.line(img1, self.tpPointsChoose[i], self.tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.imshow(self.mouseWindowName, img1)
            
            if (self.pointsCount == self.pointsMax):
                self.getRoi()
                cv2.destroyWindow(self.mouseWindowName)
        # -------------------------右键按下清除轨迹---------------
        if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
            print("right-mouse")
            self.lsPointsChoose = []
            self.tpPointsChoose = []
            self.pointsCount = 0
            # print(len(tpPointsChoose))
            for i in range(len(self.tpPointsChoose) - 1):
                # print('i', i)
                cv2.line(img, self.tpPointsChoose[i], self.tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.imshow(self.mouseWindowName, img)

if __name__ == '__main__':
    img = cv2.imread('/Volumes/Livion/Pandadataset/panda/images/scene_03_full/SEQ_03_101.jpg')
    mouse_roi = GetRoiMouse()
    mouse_roi.mouseclick()
    mouse_roi.getRoi()
    ROI = mouse_roi.ROI
    cv2.imwrite('roi.jpg', ROI)
    cv2.imshow('roi', ROI)
    cv2.waitKey(0)
