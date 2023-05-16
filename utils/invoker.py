import requests,io,json,time
import numpy as np

functions = {
    # 'keypoint' : 'http://10.1.81.24:32283',
    'keypoint' : 'http://10.1.81.24:5000',
    'yolo' : 'http://10.1.81.183:8001/'
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

