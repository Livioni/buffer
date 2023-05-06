import numpy as np
import cv2,os

class Box: 
    def __init__(self,x,y,w,h) -> None:
        self.top_left = (x,y)
        self.bottom_right = (x+w,y+h)
        self.width = w
        self.height = h
        self.father = None

class Bin:
    def __init__(self,x,y,w,h) -> None:
        self.top_left = (x,y)
        self.bottom_right = (x+w,y+h)
        self.width = w
        self.height = h
        self.items = []
    
    def insert(self, item : Box):
        self.items.append(item)

    def belongs(self, item : Box):
        if item.top_left[0] >= self.top_left[0] and item.top_left[1] >= self.top_left[1] and item.bottom_right[0] <= self.bottom_right[0] and item.bottom_right[1] <= self.bottom_right[1]:
            return True
        return False
    
    def overlap_ares(self, item : Box):
        x1 = max(self.top_left[0], item.top_left[0])
        y1 = max(self.top_left[1], item.top_left[1])
        x2 = min(self.bottom_right[0], item.bottom_right[0])
        y2 = min(self.bottom_right[1], item.bottom_right[1])
        return (x2-x1)*(y2-y1)
    
    def resize_margin(self):
        #resize the box margin to cover the all the items
        if len(self.items) == 0:
            return False
        x1 = min([item.top_left[0] for item in self.items])
        y1 = min([item.top_left[1] for item in self.items])
        x2 = max([item.bottom_right[0] for item in self.items])
        y2 = max([item.bottom_right[1] for item in self.items])
        self.top_left = (x1,y1)
        self.bottom_right = (x2,y2)
        self.width = x2-x1
        self.height = y2-y1
        return

        
def which_bin(item : Box, bins : list) -> bool:
    for bin in bins:
        if bin.belongs(item):
            bin.insert(item)
            return True
    return False

def overlap_ares(item : Box, bins : list) -> list:
    ares = []
    for bin in bins:
        ares.append(bin.overlap_ares(item))
    return ares

def bin_list_create(canvas_size : tuple = (2160,3840),lay_out : tuple = (3,3)) -> list:
    bins = []
    for i in range(lay_out[0]):
        for j in range(lay_out[1]):
            assert canvas_size[0]%lay_out[0] == 0
            assert canvas_size[1]%lay_out[1] == 0
            bins.append(Bin(i*canvas_size[1]//lay_out[1],j*canvas_size[0]//lay_out[0],canvas_size[1]//lay_out[1],canvas_size[0]//lay_out[0]))
    return bins

def bin_list_resize(bins : list):
    new_bin_list = []
    for bin in bins:
        if(bin.resize_margin()) == False:
            continue
        else:
            new_bin_list.append(bin)
    return new_bin_list

if __name__ == "__main__":
    videos_path = '/Users/livion/Documents/test_videos/Panda/4k'
    videos_list = os.listdir(videos_path)
    for video in videos_list:
        if video == '.DS_Store':
            continue
        prefix = video[0:2]
        video_path = os.path.join(videos_path,video)
        cap = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        save_path = '/Users/livion/Documents/test_videos/partitionsPY'
        index = 1
        while(True):
            if index <= 100:
                index += 1
                continue
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            if ret == False:
                break
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros_like(frame)
            bin_list = bin_list_create(canvas_size=(2160,3840),lay_out=(10,10))
            for con in contours:
                perimeter = cv2.arcLength(con,True)
                if perimeter > 0:
                    #找到一个直矩形（不会旋转）
                    x,y,w,h = cv2.boundingRect(con)
                    item = Box(x,y,w,h)
                    if which_bin(item, bin_list) == False:
                        ares = overlap_ares(item, bin_list)
                        bin_list[np.where(ares == np.max(ares))[0][0]].insert(item)
            new_bin_list = bin_list_resize(bin_list)
            for bin in new_bin_list:
                mask[bin.top_left[1]:bin.bottom_right[1],bin.top_left[0]:bin.bottom_right[0]] = frame[bin.top_left[1]:bin.bottom_right[1],bin.top_left[0]:bin.bottom_right[0]]
                # cv2.rectangle(frame,bin.top_left,bin.bottom_right,(0,255,0),3)
            print("frame " + str(index) + " done")
            if index < 10:
                save_name = 'SEQ_'+ prefix + '_00' + str(index)  + '.jpg'
            elif index < 100:
                save_name = 'SEQ_'+ prefix + '_0' + str(index)  + '.jpg'
            else:
                save_name = 'SEQ_'+ prefix +'_' + str(index)  + '.jpg'
                path_name = save_path + '/' + save_name
                cv2.imwrite(path_name,mask)
                print('save image ' + save_name)
            # cv2.imshow('mask',mask)
            # cv2.imshow('frame',frame)
            index += 1
            k = cv2.waitKey(150) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()