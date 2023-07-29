import cv2,time,logging,threading,pandas,copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.binpack import BinPack
import utils.greedypacker as greedypacker
from utils.invoker import invoke_yolo_batch_v1,invoke_yolo_batch_v3
from baselines.cost_function import Ali_function_cost_usd, Ali_idle_cost
from baselines.tools import read_response
from latecny_estimator import LatencyEstimator

class Image(object):
    '''
    The unit of image partition, which includes 
    1. image: the image itself, 
    2. created_time: the created time and, 
    3. SLO: the SLO (s)
    '''
    def __init__(self, image : cv2.Mat, created_time : float, DDL : float) -> None:
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.shape = (self.width, self.height)
        self.created_time = created_time
        # self.SLO = SLO
        self.DDL = DDL

class Pool(object):
    '''
    The queue of images, which is the input of the buffer.
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
        if image.shape[0] > self.width or image.shape[1] > self.height:
            image = self.resize_image(image, self.width, self.height)
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

    def resize_image(self, image: Image, target_width : int, target_height : int) -> Image:
        '''
        resize the image to the target size if the image is larger than the canvas size.
        1. image: the image to be resized.
        2. target_width: the target width of the image.
        3. target_height: the target height of the image.
        '''
        original_width, original_height = image.shape

        scale = min(target_height/original_height, target_width/original_width)

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_img = cv2.resize(image.image, (new_width, new_height), interpolation = cv2.INTER_AREA)

        return Image(resized_img,image.created_time,image.SLO)
        
class Queue(Pool): 
    '''
    The pool of images which are waiting to be packed.
    '''
    def __init__(self, size, height, width) -> None:
        super().__init__(size, height, width)
        self.DDL = None
        self.create_time = None
        self.efficiency = None
        self.current_result = None

    def add(self, image: Image) -> None:
        created_time = image.created_time
        ddl = image.DDL
        if self.create_time == None:
            self.create_time = created_time
        else:
            self.create_time = min(self.create_time, created_time)
        if self.DDL == None:
            self.DDL = ddl
        else:
            self.DDL = min(self.DDL, ddl)
        return super().add(image)

    def clear(self) -> None:
        self.DDL = None
        self.create_time = None
        self.current_result = None
        self.efficiency = None
        return super().clear()

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
        self.current_result = canvas_print
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
        self.current_result = canvas_print
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
        
class Table(object):
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
    def __init__(self, queue_height : int = 1000, queue_width : int = 1000, logs: bool=True, csv_record: bool = True) -> None:
        self.canvas = Queue(size=100, height=queue_height, width=queue_width)
        self.table = []
        self.slack_time = None
        self.create_time = None
        self.DDL = None
        self.timer = None
        self.old_result = None
        self.current_result = None
        self.remaining_time = None
        self.total_image = 0
        self.inference_round = 0
        self.violated_round = 0
        record_file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logs = logs
        self.csv_record = csv_record
        # init log 
        if self.logs:
            logging.basicConfig(filename='/Users/livion/Documents/GitHub/Sources/buffer/logs/logs/'+record_file_name + '.log',level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        # init csv
        if self.csv_record:
            fields = ['Timestamp', 'SLO', 'Batch Size', 'Images Number', 'Canvas efficiency', 'Remaining/Over time', \
                      'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Latency per frame (ms)','Latency per image (ms)','Cost(CNY)']
            self.data_frame = pandas.DataFrame(columns=fields)
            self.csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/logs/csv/'+record_file_name+'.csv'
            self.data_frame.to_csv(self.csv_file_path, index=False)

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
        if self.DDL is None:
            self.DDL = image.DDL
        else:
            self.DDL = min(self.DDL,image.DDL)
        self.slack_time= self.__calculate_slack_time(image)
        self.remaining_time = self.slack_time - time.time()
        if self.remaining_time > 0:
            self.timer = threading.Timer(self.remaining_time, self.__trigger)
            self.timer.start()
            return True
        else:
            if self.old_result is not None:
                self.table.pop()
                self.canvas.drop(self.canvas.index-1)
                logging.warning("Early infer due to the DDL is not satisfied")
                self.current_result = self.old_result
                t = threading.Thread(target=self.__trigger)
                t.start()
            return False

    def __calculate_slack_time(self,image : Image):
        estimate_QoS = self.__estimate_Qos(image)
        return self.DDL - estimate_QoS

    def __estimate_Qos(self, image : Image):
        self.old_result = self.current_result
        self.canvas.add(image)
        self.current_result = self.canvas.greedy_packer_solve(visualize=False)
        self.estimate_QoS = LatencyEstimator(self.canvas.width, len(self.canvas.current_result))
        return self.estimate_QoS

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
        return current_result, ddl, table, efficiency,remaining_time
    
    def __trigger(self):
        current_result, ddl, table, efficiency, remaining_time = self.record_first()
        self.clean_up()
        start_time = time.time()
        response,_ = invoke_yolo_batch_v1(current_result)
        service_time, inference_time, prepocess_time = read_response(response)
        finish_time = start_time + service_time
        if self.logs:
            self.__logs(finish_time,service_time,current_result,ddl,table,efficiency,remaining_time,start_time)
        if self.csv_record:
            self.__csv_record(finish_time, service_time, current_result, ddl, table, efficiency, remaining_time, start_time,\
                              inference_time, prepocess_time)
        return

    def __csv_record(self, finish_time : float, time_taken : float, current_result : np.ndarray, \
                     ddl : float, table : list, efficiency : float, remaining_time : float, start_time : float,\
                     inference_time : float, prepocess_time : float):
        # fields = ['Timestamp', 'SLO', 'Batch Size', 'Images Number', 'Canvas efficiency', 'Remaining/Over time', \
        #             'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Latency per frame (ms)','Latency per image (ms)','Cost(CNY)']
        whether_violated = 'No' if finish_time > ddl else 'Yes'
        remaining_over_time = ddl - finish_time
        cost = Ali_function_cost_usd(time_taken,Mem=4,CPU=2,GPU=6)
        logs = [start_time, whether_violated, len(current_result), len(table), round(efficiency,4), round(remaining_over_time,4), \
                round(inference_time*1000,4),round(prepocess_time*1000,4), round(time_taken*1000,4),round(time_taken/len(current_result),4),round(time_taken/len(table),4),cost]
        self.data_frame.loc[len(self.data_frame)]= logs 
        self.data_frame.to_csv(self.csv_file_path, index=False)
        return

    def __logs(self,finish_time : float, time_taken : float, current_result : np.ndarray, ddl : float, table : list, efficiency : float, remaining_time : float, start_time : float):
        logging.info("The DDL of this canvas: {}, The function is executed at {} ,{:.4f} ahead".format(ddl,start_time,remaining_time))
        logging.info("Canvas Batch Size: {}, Image number: {}, Canvas Avg. Efficiency: {:.4f}, Qos. {:.4f}".format(len(current_result), len(table), efficiency,time_taken))
        logging.info("The finish time is %f",finish_time)
        if finish_time >= ddl:
            logging.warning("This canvas has violated the SLO!")
            logging.warning("Overtime:{:.4f}".format(finish_time-ddl))
            self.violated_round += 1
        else:
            logging.info("Remaining time: {:.4f}".format(ddl-finish_time))
        return
    
    def show_info(self):
        logging.info("The total image number is {}, inference round is {}".format(self.total_image,self.inference_round))
        logging.info("The violated round is {}, Violate rate = {:.2f}".format(self.violated_round,self.violated_round/self.inference_round))
        return

#fixed batch size baseline algorithm implementaion
class Clipper(object):
    def __init__(self, record_file_name, batch_size: int = 8,  csv_record: bool = True) -> None:
        self.table = []
        self.batch_size = batch_size
        self.create_time = None
        self.DDL = None
        self.total_image = 0
        self.inference_round = 0
        self.violated_round = 0
        self.result = None
        # record_file_name = 'Batch_size=' + str(batch_size)+'_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_record = csv_record
        # init csv
        if self.csv_record:
            fields = ['Timestamp', 'SLO', 'Batch Size', 'Images Number', 'Size','Remaining/Over time', \
                      'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Latency per frame (ms)','Latency per image (ms)','Cost(CNY)']
            self.data_frame = pandas.DataFrame(columns=fields)
            self.csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/logs/csv/'+record_file_name+'.csv'
            self.data_frame.to_csv(self.csv_file_path, index=False)
    
    def add_batch_size(self):
        if self.batch_size >= 9:
            pass
        else:
            self.batch_size += 1
        return
    
    def drop_batch_size(self):
        self.batch_size *= 0.8
        self.batch_size = int(self.batch_size)
        return

    def push(self, image : Image):
        if self.create_time is None:
            self.create_time = image.created_time
        else:
            self.create_time = min(self.create_time,image.created_time)
        if self.DDL is None:
            self.DDL = image.DDL
        else:
            self.DDL = min(self.DDL,image.DDL)
        self.table.append(image)
        if len(self.table) >= self.batch_size:
            tables = copy.deepcopy(self.table)
            ddl = copy.deepcopy(self.DDL)
            batch_size = copy.deepcopy(self.batch_size)
            self.clean_up()
            t = threading.Thread(target=self.__auto_trigger,args=(tables,ddl,batch_size))
            t.start()
        return
    
    def trigger(self):
        if len(self.table) > 0:
            self.__auto_trigger(self.table,self.DDL)
        return
    
    def resize_table(self,tables,batch_size):
        image_width = max([image.width for image in tables])
        image_height = max([image.height for image in tables])
        result = np.zeros((batch_size, image_height, image_width, 3), dtype=np.uint8)
        for index, image in enumerate(tables):
            if image.width != image_width or image.height != image_height:
                result[index] = cv2.resize(image.image, (image_width, image_height), interpolation = cv2.INTER_AREA)
        return result

    def record_first(self):
        #clean and record all the self variable to avoid reusing
        ddl = copy.deepcopy(self.DDL)
        table = copy.deepcopy(self.table)
        return ddl, table
    
    def clean_up(self):
        self.table = []
        self.total_image += len(self.table)
        self.inference_round += 1
        self.table = []
        self.create_time = None
        self.DDL = None
        self.result = None
        return 

    def __auto_trigger(self,tables,ddl,batch_size):
        result = self.resize_table(tables,batch_size)
        start_time = time.time()
        response,_ = invoke_yolo_batch_v3(result)
        service_time, inference_time, prepocess_time = read_response(response)
        finish_time = start_time + service_time
        if self.csv_record:
            self.__csv_record(finish_time=finish_time,time_taken=service_time,current_result=result,ddl=ddl,table=tables,\
                              start_time=start_time,inference_time=inference_time,prepocess_time=prepocess_time)
        return

    def __csv_record(self, finish_time : float, time_taken : float, current_result : np.ndarray, \
                     ddl : float, table : list,  start_time : float, inference_time : float, prepocess_time : float):
        # fields = ['Timestamp', 'SLO', 'Batch Size', 'Images Number', 'Size', 'Remaining/Over time', \
        #             'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Latency per frame (ms)','Latency per image (ms)','Cost(CNY)']
        whether_violated = 'No' if finish_time > ddl else 'Yes'
        if whether_violated == 'Yes':
            self.add_batch_size()
        else:
            self.drop_batch_size()
        remaining_over_time = ddl - finish_time
        size = str(current_result.shape[1]) + 'x' + str(current_result.shape[2])
        cost = Ali_function_cost_usd(time_taken,Mem=4,CPU=2,GPU=6)
        logs = [start_time, whether_violated, len(current_result), len(table), size, round(remaining_over_time,4), \
                round(inference_time*1000,4),round(prepocess_time*1000,4), round(time_taken*1000,4), round(time_taken/len(current_result),4),round(time_taken/len(table),4),cost]
        self.data_frame.loc[len(self.data_frame)]= logs 
        self.data_frame.to_csv(self.csv_file_path, index=False)
        return

#fixed batch size baseline algorithm implementaion
class Mark(object):
    def __init__(self, record_file_name, batch_size: int = 8,  csv_record: bool = True) -> None:
        self.table = []
        self.batch_size = batch_size
        self.create_time = None
        self.DDL = None
        self.total_image = 0
        self.inference_round = 0
        self.violated_round = 0
        self.result = None
        # record_file_name = 'Batch_size=' + str(batch_size)+'_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_record = csv_record
        # init csv
        if self.csv_record:
            fields = ['Timestamp', 'SLO', 'Batch Size', 'Images Number', 'Size','Remaining/Over time', \
                      'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Latency per frame (ms)','Latency per image (ms)','Cost(CNY)']
            self.data_frame = pandas.DataFrame(columns=fields)
            self.csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/logs/csv/'+record_file_name+'.csv'
            self.data_frame.to_csv(self.csv_file_path, index=False)
    

    def push(self, image : Image):
        if self.create_time is None:
            self.create_time = image.created_time
        else:
            self.create_time = min(self.create_time,image.created_time)
        if self.DDL is None:
            self.DDL = image.DDL
        else:
            self.DDL = min(self.DDL,image.DDL)
        self.table.append(image)
        tables = copy.deepcopy(self.table)
        ddl = copy.deepcopy(self.DDL)
        batch_size = copy.deepcopy(self.batch_size)
        if len(self.table) == 1:
            self.task = threading.Timer(0.1,self.__time_trigger)
            self.task.start()
        if len(self.table) >= self.batch_size:
            t = threading.Thread(target=self.__auto_trigger,args=(tables,ddl,batch_size))
            t.start()
        return
    
    def trigger(self):
        if len(self.table) > 0:
            self.__auto_trigger(self.table,self.DDL)
        return
    
    def resize_table(self,tables,batch_size):
        image_width = max([image.width for image in tables])
        image_height = max([image.height for image in tables])
        result = np.zeros((batch_size, image_height, image_width, 3), dtype=np.uint8)
        for index, image in enumerate(tables):
            if image.width != image_width or image.height != image_height:
                result[index] = cv2.resize(image.image, (image_width, image_height), interpolation = cv2.INTER_AREA)
        return result

    def record_first(self):
        #clean and record all the self variable to avoid reusing
        ddl = copy.deepcopy(self.DDL)
        table = copy.deepcopy(self.table)
        return ddl, table
    
    def clean_up(self):
        self.table = []
        self.total_image += len(self.table)
        self.inference_round += 1
        self.table = []
        self.create_time = None
        self.DDL = None
        self.result = None
        return 

    def __time_trigger(self):
        if len(self.table) > 0:
            table = copy.deepcopy(self.table)
            ddl = copy.deepcopy(self.DDL)
            self.clean_up()
            result = self.resize_table(table,len(table))
            start_time = time.time()
            response,_ = invoke_yolo_batch_v3(result)
            service_time, inference_time, prepocess_time = read_response(response)
            finish_time = start_time + service_time
            if self.csv_record:
                self.__csv_record(finish_time=finish_time,time_taken=service_time,current_result=result,ddl=ddl,table=table,\
                                start_time=start_time,inference_time=inference_time,prepocess_time=prepocess_time)    
            return
        return
        
    def __auto_trigger(self,tables,ddl,batch_size):
        if self.task is not None:
            self.task.cancel()
        if len(tables) == 0:
            return
        self.clean_up()
        result = self.resize_table(tables,batch_size)
        start_time = time.time()
        response,_ = invoke_yolo_batch_v3(result)
        service_time, inference_time, prepocess_time = read_response(response)
        finish_time = start_time + service_time
        if self.csv_record:
            self.__csv_record(finish_time=finish_time,time_taken=service_time,current_result=result,ddl=ddl,table=tables,\
                              start_time=start_time,inference_time=inference_time,prepocess_time=prepocess_time)
        return

    def __csv_record(self, finish_time : float, time_taken : float, current_result : np.ndarray, \
                     ddl : float, table : list,  start_time : float, inference_time : float, prepocess_time : float):
        # fields = ['Timestamp', 'SLO', 'Batch Size', 'Images Number', 'Size', 'Remaining/Over time', \
        #             'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Latency per frame (ms)','Latency per image (ms)','Cost(CNY)']
        whether_violated = 'No' if finish_time > ddl else 'Yes'
        remaining_over_time = ddl - finish_time
        size = str(current_result.shape[1]) + 'x' + str(current_result.shape[2])
        cost = Ali_function_cost_usd(time_taken,Mem=4,CPU=2,GPU=6)
        logs = [start_time, whether_violated, len(current_result), len(table), size, round(remaining_over_time,4), \
                round(inference_time*1000,4),round(prepocess_time*1000,4), round(time_taken*1000,4), round(time_taken/len(current_result),4),round(time_taken/len(table),4),cost]
        self.data_frame.loc[len(self.data_frame)]= logs 
        self.data_frame.to_csv(self.csv_file_path, index=False)
        return

#time out baseline algorithm implementaion
class Timeout(object):
    def __init__(self, record_file_name, time_out: float = 0.3,  csv_record: bool = True) -> None:
        self.table = []
        self.time_out = time_out
        self.create_time = None
        self.DDL = None
        self.total_image = 0
        self.inference_round = 0
        self.violated_round = 0
        self.result = None
        self.time_out = time_out
        self.csv_record = csv_record
        # init csv
        if self.csv_record:
            fields = ['Timestamp', 'SLO', 'Batch Size', 'Shape1', 'Shape2','Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Cost(USD)']
            self.data_frame = pandas.DataFrame(columns=fields)
            self.csv_file_path = '/Users/livion/Documents/GitHub/Sources/buffer/logs/csv/'+record_file_name+'.csv'
            self.data_frame.to_csv(self.csv_file_path, index=False)
        self.run_periodically()

    def run_periodically(self):
        threading.Timer(self.time_out, self.run_periodically).start()  # 每60秒重新启动函数
        self.invoke()  # 执行你的函数

    def resize_table(self,tables):
        tables = copy.deepcopy(tables)
        image_width = max([image.width for image in tables])
        image_height = max([image.height for image in tables])
        if len(tables) * image_width * image_height > 7000000:
            return len(tables),False
        else:
            result = np.zeros((len(tables), image_height, image_width, 3), dtype=np.uint8)
            for index, image in enumerate(tables):
                if image.width != image_width or image.height != image_height:
                    result[index] = cv2.resize(image.image, (image_width, image_height), interpolation = cv2.INTER_AREA)
            return result,True
    
    def push(self, image : Image):
        if self.create_time is None:
            self.create_time = image.created_time
        else:
            self.create_time = min(self.create_time,image.created_time)
        if self.DDL is None:
            self.DDL = image.DDL
        else:
            self.DDL = min(self.DDL,image.DDL)
        self.table.append(image)
        return
    
    def clean_up(self):
        self.total_image += len(self.table)
        self.inference_round += 1
        self.table = []
        self.create_time = None
        self.DDL = None
        self.result = None
        return 
    
    def invoke(self):
        if len(self.table) > 0:
            table = copy.deepcopy(self.table)
            ddl = copy.deepcopy(self.DDL)
            self.clean_up()
            result,flag = self.resize_table(table)
            if flag == True:
                start_time = time.time()
                response,_ = invoke_yolo_batch_v3(result)
                service_time, inference_time, prepocess_time = read_response(response)
                finish_time = start_time + service_time
                if self.csv_record:
                    self.__csv_record(finish_time=finish_time,time_taken=service_time,current_result=result,ddl=ddl,\
                                    start_time=start_time,inference_time=inference_time,prepocess_time=prepocess_time)    
                return
            else:
                logs = ['here', 'NO', result, 0,0, 0, 0,0,0]
                self.data_frame.loc[len(self.data_frame)]= logs 
                self.data_frame.to_csv(self.csv_file_path, index=False)


    def __csv_record(self, finish_time : float, time_taken : float, current_result : np.ndarray, \
                     ddl : float,  start_time : float, inference_time : float, prepocess_time : float):
        # fields = ['Timestamp', 'SLO', 'Batch size', 'Size', 'Prepocess Time(ms)','Inference Time (ms)','Latency (ms)','Cost(USD)']
        whether_violated = 'No' if finish_time > ddl else 'Yes'
        size1 = str(current_result.shape[1]) 
        size2 = str(current_result.shape[2])
        cost = Ali_function_cost_usd(time_taken,Mem=4,CPU=2,GPU=6)
        logs = [start_time, whether_violated, len(current_result), size1,size2, round(prepocess_time*1000,4), \
                round(inference_time*1000,4),round(time_taken*1000,4),cost]
        self.data_frame.loc[len(self.data_frame)*2]= logs 
        self.data_frame.to_csv(self.csv_file_path, index=False)
        return

if __name__ == "__main__":
    pass


