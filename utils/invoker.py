import requests,json,time,cv2,torch
import numpy as np
from PIL import Image
from io import BytesIO

functions = {
    # 'keypoint' : 'http://10.1.81.24:32283',
    'keypoint' : 'http://10.1.81.24:5000',
    'yolo' : 'http://10.1.81.183:8001/',
    'table' : 'http://127.0.0.1:8002/'
}

def invoke_keypoint(np_data: np.ndarray):
    '''
    endpoint: the name of the function
        'debug';
        'keypoint'
    input_data: the image data
    '''
    endpoint_url = functions['keypoint']
  
    batch_data = np_data.tolist()
    shape = np_data.shape

    payload = {'data': batch_data,'shape':shape}
    payload_str = json.dumps(payload)
    # Send the HTTP POST request
    # response = requests.post(endpoint_url, files={'file': bytes_data,'size': size_data},timeout=30)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(endpoint_url, headers=headers, data=payload_str,timeout=30)
    return response

def invoke_yolo(np_data: np.ndarray):
    '''
    invoke yolo function through json data
    '''
    endpoint_url = functions['yolo']
    type_rq = 'batch_inference/'
    shape = np_data.shape
    data = {
    "data": np_data.tolist(),
    "shape": shape
    }   
    start = time.time()
    response = requests.post(endpoint_url+type_rq, json=data)
    end = time.time()
    print("Time taken: ", end-start)
    return response.json()

def invoke_yolo_single(image_path: str):
    '''
    invoke yolo function through image path
    '''
    endpoint_url = functions['yolo']
    type_rq = 'img_object_detection_to_img/'
    files = {'file': open(image_path, 'rb')}
    start = time.time()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.time()
    time_taken = end-start
    img = Image.open(BytesIO(response.content)) 
    return time_taken

def invoke_yolo_batch_v1(np_data : np.ndarray):
    '''
    invoke yolo function through a batch of images with np.ndarray format
    '''
    endpoint_url = functions['yolo']
    type_rq = 'uploadfiles/'
    files = []
    for index, img in enumerate(np_data):
        ret, img_encode = cv2.imencode('.jpg', img)
        f4 = BytesIO(img_encode ) # 这样可以直接转换，无需再转 img_encode.tostring()
        files.append(('files', ('image'+str(index)+'.jpg', f4.getvalue(), 'image/jpeg')))
    start = time.time()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.time()
    time_taken = end-start
    return response.json(),time_taken

def invoke_yolo_batch(images_list : list):
    '''
    invoke yolo function through a list of images with image path
    '''    
    endpoint_url = functions['yolo']
    type_rq = 'uploadfiles/'
    files = [('files', (open(file, 'rb'))) for file in images_list]
    start = time.time()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.time()
    print("Time taken: ", end-start)
    return response.json()

def push_to_table(np_data: np.ndarray, delay_time :float, SLO: float=1.0):
    endpoint_url = functions['table']
    type_rq = 'uploadimage/'
    ret, img_encode = cv2.imencode('.jpg', np_data)
    f4 = BytesIO(img_encode) 
    created_time = time.time()
    time.sleep(delay_time)
    multipart_form_data = {
        'file': (f4.getvalue()),
        'created_time': (None, str(created_time)),
        'slo': (None, str(SLO)),
    }
    response = requests.post(endpoint_url+type_rq, files=multipart_form_data)
    return response.json()

if __name__ == "__main__":
    image = '/Users/livion/Documents/test_videos/partitions_01/101_1.jpg'
    time_taken = invoke_yolo_single(image)
    print(time_taken)