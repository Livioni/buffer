import cv2
import datetime as dt
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from utils.invoker import invoke_keypoint
from utils.binpack import BinPack


class Image:
    def __init__(self, image : cv2.Mat) -> None:
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.shape = (self.width, self.height)

#write a class to buffer the image
class Canvas: #wait to be transfered to function
    def __init__(self, size, height, width, time_out: int = 5):
        self.size = size
        self.height = height
        self.width = width
        #create buffers to save the images
        self.buffer = np.zeros((self.size, self.height, self.width, 3), dtype=np.uint8)
        self.image_name = []
        self.index = 0
        self.time_out = time_out
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.send_buffer, 'interval', seconds = self.time_out)
        # self.scheduler.start()

    def send_buffer(self):
        if self.index != 0:
            self.is_full()
        else:
            print("Buffer",self.size, dt.datetime.now())
        return
        
    def is_full(self):
        #invoke the function
        ret = invoke_keypoint(self.buffer)
        print("Function returned: ", ret.text)
        return self.delete()

    def add(self, image : Image):

        self.buffer[self.index] = image.image
        self.image_name.append(image.image_name)
        self.index += 1
        if self.index == self.size:
            self.is_full()
        return True

    def get(self):
        return self.buffer
    
    def get_image(self, index):
        return self.buffer[index]
    
    def delete(self):
        self.buffer = np.zeros((self.size, self.height, self.width, 3), dtype=np.uint8)
        self.image_name = []
        self.index = 0

    def __resize_img_keep_ratio(self, img, target_size: list):
        old_size= img.shape[0:2]
        ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i*ratio) for i in old_size])
        img = cv2.resize(img,(new_size[1], new_size[0]))
        pad_w = target_size[1] - new_size[1]
        pad_h = target_size[0] - new_size[0]
        top,bottom = pad_h//2, pad_h-(pad_h//2)
        left,right = pad_w//2, pad_w -(pad_w//2)
        img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
        return img_new

class Queue: #wait to be packing
    def __init__(self, size) -> None:
        self.size = size
        self.queue = []
        self.index = 0

    def add(self, image : Image):
        if self.index == self.size:
            self.clear()
        self.queue.append(image)
        self.index += 1
        return True
    
    def insert(self, *items: Image):
        for item in items:
            self.add(item)

    def clear(self):
        self.queue = []
        self.index = 0

    def solve(self, canvas : Canvas, heuristic : str = 'best_fit'):
        BINPACK = BinPack(bin_size=(canvas.width,canvas.height))

        for item in self.queue:
            BINPACK.insert(item.shape, heuristic = heuristic)
        result = BINPACK.print_stats()
        # BINPACK.visualize_packing(result)
        canvas_print = np.zeros((len(result)-1, canvas.height, canvas.width, 3), dtype=np.uint8)
        for i in range(len(result)-1):
            for item in result[i]['items']:
                locationx = item[0].x
                locationy = item[0].y
                image_width = item[1].width
                image_height = item[1].height
                for img in self.queue:
                    if img.width == image_width and img.height == image_height:
                        canvas_print[i,locationy:locationy+image_height, locationx:locationx+image_width] = img.image
                        break
        # for i in range(len(result)-1):
        #     cv2.imshow("Canvas", canvas_print[i])
        #     cv2.waitKey()
        self.clear()
        return canvas_print


if __name__ == "__main__":
    im1 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image43_10.jpg')
    im2 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image43_11.jpg')
    im3 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image43_12.jpg')
    im4 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_0.jpg')
    im5 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_1.jpg')
    im6 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_2.jpg')
    im7 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_3.jpg')
    im8 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_4.jpg')
    im9 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_5.jpg')
    im10 = cv2.imread('/Users/livionmbp/Documents/test_videos/partitions/image44_6.jpg')
    img1 = Image(im1)
    img2 = Image(im2)
    img3 = Image(im3)
    img4 = Image(im4)
    img5 = Image(im5)
    img6 = Image(im6)
    img7 = Image(im7)
    img8 = Image(im8)
    img9 = Image(im9)
    img10 = Image(im10)
    boxs = Queue(10)
    boxs.insert(img1, img2, img3, img4, img5, img6, img7, img8, img9, img10)
    canvas = Canvas(3, 300, 300)
    boxs.solve(canvas)


