import cv2
import numpy as np
from utils.invoker import invoke_keypoint

#write a class to buffer the image
class Buffer:
    def __init__(self, size, height, width):
        self.size = size
        self.height = height
        self.width = width
        #create buffers to save the images
        self.buffer = np.zeros((self.size, self.height, self.width, 3), dtype=np.uint8)
        self.image_name = []
        self.index = 0

    def is_full(self):
        #invoke the function
        ret = invoke_keypoint(self.buffer)
        print("Function returned: ", ret.text)
        return self.delete()

    def add(self, image,image_name):
        if image.shape[0] != self.height or image.shape[1] != self.width:
            image = self.__resize_img_keep_ratio(image,[self.height,self.width])
        self.buffer[self.index] = image
        self.image_name.append(image_name)
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

if __name__ == "__main__":
    buffer = Buffer(2, 640, 480)
    img1 = cv2.imread('/Users/livion/Documents/test_videos/input/image3_4.jpg')
    img2 = cv2.imread('/Users/livion/Documents/test_videos/input/image3_9.jpg')
    buffer.add(img1,"image1")
    buffer.add(img2,"image2")
    print("here")
