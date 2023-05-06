import cv2,time
import numpy as np


#经典的测试视频
cap = cv2.VideoCapture('/Users/livion/Documents/test_videos/Panda/4k/04_Primary_School.mp4')
#形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()
start_time = time.time()
counter = 0
dilate = True
dilate_size = 0 
save = False
save_path = '/Users/livion/Documents/test_videos/partitionsPY'
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
print("视频平均帧率",fps)


def partitions(img: cv2.Mat,x,y,w,h):
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    w = img.shape[1] - x if x + w > img.shape[1] else w
    h = img.shape[0] - y if y + h > img.shape[0] else h
    sub_image = img[y:y+h, x:x+w]
    return sub_image


while(True):
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)  # 窗口大小
    fgmask = fgbg.apply(frame)
    if ret == False:
        break
    #形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #寻找视频中的轮廓
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    index = 0
    mask = np.zeros_like(frame)
    for con in contours:
        #计算各轮廓的周长
        perimeter = cv2.arcLength(con,True)
        if perimeter > 100:
            #找到一个直矩形（不会旋转）
            x,y,w,h = cv2.boundingRect(con)
            mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
            #画出这个矩形
            if dilate:
                x = x - dilate_size if x - dilate_size > 0 else 0
                y = y - dilate_size if y - dilate_size > 0 else 0
                w = w + 2*dilate_size if w + dilate_size < frame.shape[1] else frame.shape[1]
                h = h + 2*dilate_size if h + dilate_size < frame.shape[0] else frame.shape[0]
                if not save: cv2.rectangle(frame,(x,y),(x+w,y+h),(19,33,207),2)
            else:
                if not save: cv2.rectangle(frame,(x,y),(x+w,y+h),(19,33,207),2) 
            if save:
                save_name = 'image' + str(counter) + '_' + str(index) + '.jpg'
                sub_image = partitions(frame, x,y,w,h)
                path_name = save_path + '/' + save_name
                cv2.imwrite(path_name,sub_image)
                index += 1

    counter += 1
    if (time.time() - start_time) != 0:  # 实时显示帧数
        cv2.putText(frame, "FPS {0}".format(float('%.1f' % (1 / (time.time() - start_time)))), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    # cv2.imshow('fgmask', fgmask)
    

    #计时器
    print("FPS: ", 1 / (time.time() - start_time))        
    start_time = time.time()

    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

