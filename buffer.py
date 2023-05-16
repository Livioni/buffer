import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.binpack import BinPack
import utils.greedypacker as greedypacker

class Image:
    def __init__(self, image : cv2.Mat) -> None:
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.shape = (self.width, self.height)

class Queue: #wait to be packing
    def __init__(self, size, height, width) -> None:
        self.size = size
        self.height = height
        self.width = width
        self.efficiency = None
        self.queue = []
        self.index = 0

    def add(self, image : Image):
        if self.index == self.size:
            print("Warning! : Queue is full")
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

    def calculate_efficiency(self, result : np.ndarray) -> float:

        efficiency_pool = []
        for i in range(len(result)-1):
            efficiency_pool.append(result[i]['efficiency'])
            print("Bin", i, "efficiency: ", result[i]['efficiency'])
        self.efficiency = np.mean(efficiency_pool)
        return self.efficiency

    def bin_pack_solve(self, heuristic : str = 'next_fit',visualize : bool = False) -> np.ndarray:
        BINPACK = BinPack(bin_size=(self.width,self.height),sorting=True)
        for item in self.queue:
            BINPACK.insert(item.shape, heuristic = heuristic)
        result = BINPACK.print_stats()
        canvas_print = np.zeros((len(result)-1, self.height,self.width, 3), dtype=np.uint8)
        efficiency_pool = []
        for i in range(len(result)-1):
            efficiency_pool.append(result[i]['efficiency'])
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
        self.efficiency = np.mean(efficiency_pool)
        if visualize:
            for i in range(len(result)-1):
                cv2.imshow("Canvas", canvas_print[i])
                cv2.waitKey()
        self.clear()
        return canvas_print
    
    def greedy_packer_solve(self, visualize : bool = False, pack_algo : str ='guillotine', heuristic : str = 'best_shortside') -> np.ndarray:
        GreedyPacker = greedypacker.BinManager(self.height, self.width, pack_algo=pack_algo, heuristic=heuristic, wastemap=False, rotation=False)
        for item in self.queue:
            image_item = greedypacker.Item(item.width, item.height)
            GreedyPacker.add_items(image_item)
        GreedyPacker.execute()
        result = GreedyPacker.bins
        canvas_print = np.zeros((len(result), self.height, self.width, 3), dtype=np.uint8)
        efficiency_pool = []
        for i in range(len(result)):
            canvas_area = self.height * self.width
            total_area = 0
            current_queue = self.queue.copy()
            for item in result[i].items:
                image_area = item.width * item.height
                total_area += image_area
                for index, img in enumerate(current_queue):
                    if (img.width == item.width and img.height == item.height):
                        canvas_print[i,item.y:item.y+item.height, item.x:item.x+item.width] = img.image
                        current_queue.pop(index)
                        break
            
            efficiency_pool.append(total_area/canvas_area)
        self.efficiency = np.mean(efficiency_pool)
        if visualize:
            self.visualize_packing(result)
            for i in range(len(result)):
                cv2.imshow("Canvas", canvas_print[i])
                cv2.waitKey()
        return canvas_print
    
    def visualize_packing(self,result) -> None:
        colors = plt.cm.tab20.colors
        colors_index = 0
        for sheet in result:
            fig, ax = plt.subplots()   
            ax.set_title('Bin Packing Visualization')
            for item in sheet.items:
                rect = plt.Rectangle((item.x, item.y), item.width, item.height, color=colors[colors_index % len(colors)])
                ax.add_patch(rect)
                colors_index += 1
            plt.xlim(0, self.width)
            plt.ylim(0, self.height)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        return
        



