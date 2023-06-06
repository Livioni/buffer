import requests,json,time,cv2
import numpy as np
from io import BytesIO

functions = {
    # 'keypoint' : 'http://10.1.81.24:32283',
    'keypoint' : 'http://10.1.81.24:5000',
    # 'yolo' : 'http://10.1.81.183:8001/', #ray
    # 'yolo' : 'http://10.106.5.35:8000/',   #asd
    'yolo':'http://yolov-inference-yolovx-idcixdzubx.cn-hangzhou.fcapp.run/', #ali cloud
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

def invoke_yolo_single(np_data : np.ndarray):
    '''
    invoke yolo function through image path
    '''
    endpoint_url = functions['yolo']
    type_rq = 'single_image/'
    files = []
    for index, img in enumerate(np_data):
        ret, img_encode = cv2.imencode('.jpg', img)
        f4 = BytesIO(img_encode ) # 这样可以直接转换，无需再转 img_encode.tostring()
        files.append(('files', ('image'+str(index)+'.jpg', f4.getvalue(), 'image/jpeg')))
    start = time.perf_counter()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.perf_counter()
    time_taken = end-start
    return response.json(),time_taken

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
    start = time.perf_counter()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.perf_counter()
    time_taken = end-start
    return response.json(),time_taken

def invoke_yolo_batch_v2(np_data : np.ndarray):
    '''
    invoke yolo function through a batch of images with np.ndarray format
    '''
    endpoint_url = functions['yolo']
    type_rq = 'full/'
    files = []
    for index, img in enumerate(np_data):
        ret, img_encode = cv2.imencode('.jpg', img)
        f4 = BytesIO(img_encode ) # 这样可以直接转换，无需再转 img_encode.tostring()
        files.append(('files', ('image'+str(index)+'.jpg', f4.getvalue(), 'image/jpeg')))
    start = time.perf_counter()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.perf_counter()
    time_taken = end-start
    return response.json(),time_taken

def invoke_yolo_batch(images_list : list):
    '''
    invoke yolo function through a list of images with image path
    '''    
    endpoint_url = functions['yolo']
    type_rq = 'uploadfiles/'
    files = [('files', (open(file, 'rb'))) for file in images_list]
    start = time.perf_counter()
    response = requests.post(endpoint_url+type_rq, files=files)
    end = time.perf_counter()
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
    import os
    image = '/Volumes/Livion/Pandadataset/图片/4k/SEQ_01_001.jpg'
    network_bandwidth = 100 # 10Mbps
    upload_byte_per_second = network_bandwidth * 1000 * 1000 / 8
    file_size = os.path.getsize(image)
    transmission_time = file_size/upload_byte_per_second # 假设带宽为10Mbps
    image = cv2.imread(image)
    image_numpy = np.array(image)
    new_image = np.expand_dims(image_numpy, axis=0)
    response,time_taken = invoke_yolo_single(new_image)
    response_list = response.split(' ')
    service_time = float(response_list[1][:-1])
    inference_time = float(response_list[3][:-1])
    prepocess_time = float(response_list[5][:-1])
    print('Service time',service_time)
    print('Inference time',inference_time)
    print('Preprocess time',prepocess_time)
    print("Transmission time",transmission_time)
    print("Total time by calculation:",service_time+transmission_time)
    # _,time_taken = invoke_yolo_batch_v1(image)
    print('Total time:',time_taken)