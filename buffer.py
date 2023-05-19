import cv2,time,logging,threading
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.binpack import BinPack
import utils.greedypacker as greedypacker
from utils.invoker import invoke_yolo_batch_v1

class Image:
    '''
    The unit of image partition, which includes 
    1. image: the image itself, 
    2. created_time: the created time and, 
    3. SLO: the SLO (s)
    '''
    def __init__(self, image : cv2.Mat, created_time : float, SLO : float = 0.4) -> None:
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.shape = (self.width, self.height)
        self.created_time = created_time
        self.SLO = SLO

class Queue: #wait to be packing
    '''
    The queue of images, which is the input of the buffer. We bin pack the images and POST them in a batch to the server.
    1. size: the size of the queue. (We set it to a large number to avoid the overflow)
    2. queue: list of Image
    3. index: the number of Image
    4. height, width: size of canvas
    '''
    def __init__(self, size, height, width) -> None:
        self.size = size
        self.height = height
        self.width = width
        self.efficiency = None
        self.queue = []
        self.index = 0

    def add(self, image : Image) -> None:
        '''
        Add a Image to the queue.
        '''
        if self.index == self.size:
            print("Warning! : Queue is full")
            self.clear()
        self.queue.append(image)
        self.index += 1
        return 
    
    def insert(self, *items: Image) -> None:
        '''
        Add a list of Image to the queue.
        '''
        for item in items:
            self.add(item)
        return

    def clear(self) -> None:
        '''
        clear the queue.
        '''
        self.queue = []
        self.index = 0
        return

    def drop(self, index) -> None:
        '''
        Drop one specific image in the queue.
        1. index: the index of the image to be dropped.
        '''
        self.queue.pop(index)
        self.index -= 1
        return

    def bin_pack_solve(self, heuristic : str = 'next_fit',visualize : bool = False) -> np.ndarray:
        '''
        bin pack the images in the queue using binpack library () and return the canvas.
        1. heuristic: the heuristic of binpack library.
        2. visualize: whether to visualize the packing process.
        '''
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
        '''
        bin pack the images in the queue using greedy_packer library (https://github.com/solomon-b/greedypacker) and return the canvas. 
        1. visualize: whether to visualize the packing process.
        2. pack_algo: the packing algorithm of greedy_packer library.
        3. heuristic: the heuristic of greedy_packer library.
        '''
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
        '''
        Visulize the packing process.
        1. result: the result of the bin packing solver.
        '''
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
        

class Table:
    '''
    A table to store the information of the batch, which determines when to post the batch and drop a Image to a new table.
    Algorithm:
        The greedy packer will calculate the batch size of the canvas once a new image is arrived, then we get the Slcak Time by the following equation.
            Slack Time = SLO - Qos_per_frame x Batch_Size
        Qos per frame is a conservative estimate of the inference of a canvas, for example, a canvas with 2000x2000 size will take max 0.28s to inference by yolov8x in our environment.
        Then we set the timer to be the DDL(the earliest image's deadline represents the DDL of the whole canvas) - Slack Time, and if the timer is triggered, we will post the batch to the server.
        Due to the Qos_per_frame is a conservative estimation, We can reduce the violate rate.
    1. canvas: the Queue of the table.
    2. create_time: the earliest file's arrival time.
    3. table: a list to save Images.
    4. Qos_per_batch: a conservative estimate of the inference of a canvas.
    5. DDL: the earliest image's deadline
    '''
    def __init__(self, queue_height : int = 1000, queue_width : int = 1000, Qos_per_batch : float = 0.28) -> None:
        self.canvas = Queue(size=100, height=queue_height, width=queue_width)
        self.table = []
        self.Qos_per_batch = Qos_per_batch
        self.slack_time = None
        self.create_time = None
        self.DDL = None
        self.timer = None
        self.old_result = None
        self.current_result = None
        self.remaining_time = None
        self.step_flag = False
        self.total_image = 0
        self.inference_round = 0
        self.violated_round = 0
        # init log
        log_file_name = datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
        logging.basicConfig(filename='/Users/livion/Documents/GitHub/Sources/buffer/logs/'+log_file_name,level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    def __repr__(self) -> str:
        return "Table: {}".format(self.table)

    def push(self, image : Image) -> bool:
        '''
        Push a Image to the table, the table will calculate the slack time and set a timer to invoker function.
        '''
        if self.timer is not None:
            self.timer.cancel()
        self.table.append(image)
        if self.create_time is None:
            self.create_time = image.created_time
        else:
            self.create_time = min(self.create_time,image.created_time)
        self.DDL = self.create_time + image.SLO
        self.slack_time= self.__calculate_slack_time(image)
        self.remaining_time = self.slack_time - time.time()
        if self.remaining_time > 0:
            self.timer = threading.Timer(self.remaining_time, self.__trigger)
            self.timer.start()
            return True
        else:
            if self.step_flag:
                self.table.pop()
                self.canvas.drop(self.canvas.index-1)
                logging.warning("Drop a image due to the DDL is not satisfied")
                self.current_result = self.old_result
                t = threading.Thread(target=self.__trigger)
                t.start()
                return False
            else:
                t = threading.Thread(target=self.__trigger)
                t.start()
                return True


    def __calculate_slack_time(self,image : Image):
        estimate_QoS = self.__estimate_Qos(image)
        return self.DDL - estimate_QoS

    def __estimate_Qos(self, image : Image):
        self.old_result = self.current_result
        self.canvas.add(image)
        self.current_result = self.canvas.greedy_packer_solve(visualize=False)
        self.estimate_QoS = len(self.current_result) * self.Qos_per_batch
        self.__more_than_one()
        return self.estimate_QoS

    def __more_than_one(self):
        if self.old_result is not None:
            self.step_flag = True if (len(self.current_result)-len(self.old_result)) > 0 else False
        else:
            self.step_flag = False
        return 

    def clean_up(self):
        self.canvas.clear()
        self.total_image += len(self.table)
        self.inference_round += 1
        self.table = []
        self.slack_time = None
        self.create_time = None
        self.DDL = None
        self.timer = None
        self.current_result = None
        self.remaining_time = None
        self.old_result = None
        self.step_flag = False
        return 
    
    def record_first(self):
        #clean and record all the self variable to avoid reusing
        current_result = self.current_result
        ddl = self.DDL
        table = self.table
        efficiency = self.canvas.efficiency
        remaining_time = self.remaining_time
        self.clean_up()
        return current_result, ddl, table, efficiency,remaining_time
    
    def __trigger(self):
        current_result, ddl, table, efficiency, remaining_time = self.record_first()
        start_time = time.time()
        response,time_taken = invoke_yolo_batch_v1(current_result)
        finish_time = time.time()
        self.__logs(finish_time,time_taken,current_result,ddl,table,efficiency,remaining_time,start_time)
        return

    def __logs(self,finish_time : float, time_taken : float, current_result : np.ndarray, ddl : float, table : list, efficiency : float, remaining_time : float, start_time : float):
        logging.info("The DDL of this canvas: {}, The function is executed at {} ,{} ahead".format(ddl,start_time,remaining_time))
        logging.info("Canvas Batch Size: {}, Image number: {}, Canvas Avg. Efficiency: {}, Qos. {}".format(len(current_result), len(table), efficiency,time_taken))
        logging.info("The finish time is %f",finish_time)
        if finish_time >= ddl:
            logging.warning("This canvas has violated the SLO!")
            logging.warning("Overtime:%f",finish_time-ddl)
            self.violated_round += 1
        else:
            logging.info("Remaining time: %f",ddl-finish_time)
        return
    
    def show_info(self):
        logging.info("The total image number is {}, inference round is {}".format(self.total_image,self.inference_round))
        logging.info("The violated round is {}, Violate rate = {}".format(self.violated_round,self.violated_round/self.inference_round))
        return

    










        


    




