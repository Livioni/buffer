import numpy as np
import cv2,os,time
from scipy.stats import kstest
import yaml,sys
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from buffer import Table, Image
network_bandwidth = 1000 #kbps

def read_yaml_all(yaml_path):
    with open(yaml_path,"r",encoding="utf-8") as f:
        data=yaml.load(f,Loader=yaml.FullLoader)
        return data

    
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
        if x1 >= x2 or y1 >= y2:
            return 0
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
    
    def central_point(self):
        #return the central point of the box
        center = {}
        for index, item in enumerate(self.items):
            center[index] = ((item.top_left[0]+item.bottom_right[0])/2,(item.top_left[1]+item.bottom_right[1])/2)
        return center


    def _3_sigma(self, point : Box):
        #Get the mean and standard deviation of the coordinates
        center = self.central_point()
        if not center:
            return True
        x_mean = np.mean([center[index][0] for index in center])
        y_mean = np.mean([center[index][1] for index in center])
        x_std = np.std([center[index][0] for index in center])
        y_std = np.std([center[index][1] for index in center])
        # judge whether to obey normal
        x_norm = kstest([center[index][0] for index in center], 'norm')
        y_norm = kstest([center[index][1] for index in center], 'norm')
        if x_norm[1] >= 0.05 or y_norm[1] >= 0.05:
            raise Exception('The distribution of the coordinates does not obey the normal distribution')
        # judge whether the point is within 3 sigma
        x_center = (point.top_left[0]+point.bottom_right[0])/2
        y_center = (point.top_left[1]+point.bottom_right[1])/2
        if abs(x_center-x_mean) <= 3*x_std and abs(y_center-y_mean) <= 3*y_std:
            return True
        return False

def belongs_to_which_bin(item : Box, bins : list):
    for index,bin in enumerate(bins):
        if bin.belongs(item):
            return index
    return False

def overlap_ares(item : Box, bins : list) -> list:
    ares = []
    for bin in bins:
        ares.append(bin.overlap_ares(item))
    return ares

def bin_list_create(canvas_size : tuple = (2160,3840),lay_out : tuple = (3,3)) -> list:
    bins = []
    for i in range(lay_out[1]):
        for j in range(lay_out[0]):
            assert canvas_size[0] % lay_out[0] == 0
            assert canvas_size[1] % lay_out[1] == 0
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
    configuration_path = '/Users/livion/Documents/GitHub/Sources/buffer/utils/background/configuration.yaml'
    configration = read_yaml_all(configuration_path)
    videos_list = []
    # table1 = Table(1000,1000,0.165)
    # time.sleep(1)
    # table2 = Table(1000,1000,0.165)
    # switch = False
    for num,value in configration.items():
        cap = cv2.VideoCapture(value['path'])
        fgbg = cv2.createBackgroundSubtractorMOG2()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        ##########configuration##########
        bin_lay_out = tuple((int(value['lay_out'][0]),int(value['lay_out'][2])))
        threshold = value['threshold']
        save_path = value['save_path']
        partitions_save_path = value['partitions_save_path']
        prefix = value['prefix']
        #############filter##############
        hand_mask = np.zeros((value['img_size'][1], value['img_size'][0]), dtype=np.uint8)
        x_data = np.array(value['x_data'])
        y_data = np.array(value['y_data'])
        pts = np.vstack((x_data, y_data)).astype(np.int32).T
        cv2.fillPoly(hand_mask, [pts], (255), 8, 0)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        index = 1
        ################run##############
        while(True):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.bitwise_and(frame, frame, mask=hand_mask)
            fgmask = fgbg.apply(frame)
            if index <= 100:
                index += 1
                continue
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros_like(frame)
            bin_list = bin_list_create(canvas_size=(2160,3840),lay_out=bin_lay_out)
            # draw bin on the frame
            # for bin in bin_list:
            #     cv2.rectangle(frame,bin.top_left,bin.bottom_right,(0,0,255),2)
            for con in contours:
                perimeter = cv2.arcLength(con,True)
                if perimeter > threshold:
                    #找到一个直矩形（不会旋转）
                    x,y,w,h = cv2.boundingRect(con)
                    # cv2.rectangle(frame,(x,y),(x+w,y+h),(207,33,20),2) 
                    item = Box(x,y,w,h)
                    result = belongs_to_which_bin(item, bin_list)
                    if result != False:
                        bin_list[result].insert(item)
                    else:
                        ares = overlap_ares(item, bin_list)
                        bin_list[np.where(ares == np.max(ares))[0][0]].insert(item)

            new_bin_list = bin_list_resize(bin_list)
            for bin in new_bin_list:
                mask[bin.top_left[1]:bin.bottom_right[1],bin.top_left[0]:bin.bottom_right[0]] = frame[bin.top_left[1]:bin.bottom_right[1],bin.top_left[0]:bin.bottom_right[0]]
                cv2.rectangle(mask,bin.top_left,bin.bottom_right,(0,255,0),3)
            print("frame " + str(index) + " done")
            if index < 10:
                save_name = 'SEQ_'+ prefix + '_00' + str(index)  + '.jpg'
            elif index <= 100:
                save_name = 'SEQ_'+ prefix + '_0' + str(index)  + '.jpg'
            else:
                save_name = 'SEQ_'+ prefix +'_' + str(index)  + '.jpg'
                path_name = save_path + '/' + save_name
                cv2.imwrite(path_name,mask)
                print('save image ' + save_name)
                if not os.path.exists(partitions_save_path):
                    os.mkdir(partitions_save_path)
                for id,bin_area in enumerate(new_bin_list):
                    cv2.imwrite(partitions_save_path + '/' + str(index) + '_' + str(id) + '.jpg', mask[bin_area.top_left[1]:bin_area.bottom_right[1],bin_area.top_left[0]:bin_area.bottom_right[0]])
                    # mat_format = mask[bin_area.top_left[1]:bin_area.bottom_right[1],bin_area.top_left[0]:bin_area.bottom_right[0]]
                    # file_size = os.path.getsize(partitions_save_path + '/' + str(index) + '_' + str(id) + '.jpg')
                    # delay_time = file_size / (network_bandwidth * 1000)
                    # image = Image(mat_format,time.time(),1)
                    # time.sleep(delay_time)
                    # if switch == False:
                    #     if table1.push(image) == False:
                    #         table2.push(image)
                    #         switch = True
                    # else:
                    #     if table2.push(image) == False:
                    #         table1.push(image)
                    #         switch = False
            # cv2.imshow('mask',mask)
            # cv2.imshow('frame',frame)
            index += 1
            k = cv2.waitKey(150) & 0xff
            if k == 27:
                break
        # table1.show_info()
        # table2.show_info()
        cap.release()
        cv2.destroyAllWindows()