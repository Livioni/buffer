import cv2
import numpy as np
from utils.invoker import invoke_debug

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
        for image in self.buffer:
            ret = invoke_debug(image)
        return self.delete()

    def add(self, image,image_name):
        if image.shape[0] != self.height or image.shape[1] != self.width:
            image = cv2.resize(image, (self.width, self.height))
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


