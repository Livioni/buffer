import cv2
import datetime as dt
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib.pyplot as plt
from utils.invoker import invoke_keypoint
from utils.binpack import BinPack
import utils.greedypacker as greedypacker


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

    def drop(self, index):
        self.queue.pop(index)

    def bin_pack_calculate_efficiency(self, canvas : Canvas, heuristic : str = 'best_fit') -> float:
        BINPACK = BinPack(bin_size=(canvas.width,canvas.height))
        for item in self.queue:
            BINPACK.insert(item.shape, heuristic = heuristic)
        result = BINPACK.print_stats()
        efficiency_pool = []
        for i in range(len(result)-1):
            efficiency_pool.append(result[i]['efficiency'])
            print("Bin", i, "efficiency: ", result[i]['efficiency'])
        self.efficiency = np.mean(efficiency_pool)
        return self.efficiency


    def bin_pack_solve(self, canvas : Canvas, heuristic : str = 'next_fit',visualize : bool = False) -> np.ndarray:
        BINPACK = BinPack(bin_size=(canvas.width,canvas.height),sorting=True)
        for item in self.queue:
            BINPACK.insert(item.shape, heuristic = heuristic)
        result = BINPACK.print_stats()
        canvas_print = np.zeros((len(result)-1, canvas.height, canvas.width, 3), dtype=np.uint8)
        for i in range(len(result)-1):
            for item in result[i]['items']:
                locationx = item[0].x
                locationy = item[0].y
                image_width = item[1].width
                image_height = item[1].height
                for index, img in enumerate(self.queue):
                    if img.width == image_width and img.height == image_height:
                        canvas_print[i,locationy:locationy+image_height, locationx:locationx+image_width] = img.image
                        self.drop(index)
                        break
        if visualize:
            for i in range(len(result)-1):
                cv2.imshow("Canvas", canvas_print[i])
                cv2.waitKey()
        self.clear()
        return canvas_print
    
    def greedy_packer_solve(self,canvas : Canvas, visualize : bool = False, pack_algo : str ='guillotine', heuristic : str = 'best_shortside') -> np.ndarray:
        GreedyPacker = greedypacker.BinManager(canvas.height, canvas.width, pack_algo=pack_algo, heuristic=heuristic, wastemap=False, rotation=False)
        for item in self.queue:
            image_item = greedypacker.Item(item.width, item.height)
            GreedyPacker.add_items(image_item)
        GreedyPacker.execute()
        result = GreedyPacker.bins
        canvas_print = np.zeros((len(result), canvas.height, canvas.width, 3), dtype=np.uint8)
        for i in range(len(result)):
            for item in result[i].items:
                for index, img in enumerate(self.queue):
                    if (img.width == item.width and img.height == item.height):
                        canvas_print[i,item.y:item.y+item.height, item.x:item.x+item.width] = img.image
                        self.drop(index)
                        break
                    # elif (img.width == item.height and img.height == item.width):
                    #     canvas_print[i,item.y:item.y+item.width, item.x:item.x+item.height] = img.image
                    #     self.drop(index)
                    #     break
        if visualize:
            self.visualize_packing(result,canvas)
            for i in range(len(result)):
                cv2.imshow("Canvas", canvas_print[i])
                cv2.waitKey()
        return canvas_print
    
    def visualize_packing(self,result,canvas : Canvas) -> None:
        colors = plt.cm.tab20.colors
        colors_index = 0
        for sheet in result:
            fig, ax = plt.subplots()   
            ax.set_title('Bin Packing Visualization')
            for item in sheet.items:
                rect = plt.Rectangle((item.x, item.y), item.width, item.height, color=colors[colors_index % len(colors)])
                ax.add_patch(rect)
                colors_index += 1
            plt.xlim(0, canvas.width)
            plt.ylim(0, canvas.height)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        return
        

if __name__ == "__main__":
    image_list = np.linspace(0,9,10)
    boxs = Queue(10)
    for i in image_list:
        im_path = '/Users/livion/Documents/test_videos/partitions/image100_' + str(int(i)) + '.jpg'
        img = Image(cv2.imread(im_path))
        boxs.add(img)
    canvas = Canvas(3, 500, 500)
    # boxs.bin_pack_solve(canvas, visualize=True)
    boxs.greedy_packer_solve(canvas, visualize=True)
    # boxs.calculate_efficiency(canvas)
    # boxs.solve(canvas,visualize=True)


