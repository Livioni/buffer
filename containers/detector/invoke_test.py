import numpy as np
import requests,io,json
import time

functions = {
    # 'keypoint' : 'http://10.1.81.24:32283',
    'keypoint' : 'http://10.1.81.183:6000',
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

def invoke_detector(batch_size,canvas_size : tuple):
    random_images = np.random.randint(0,255,(batch_size,canvas_size[0],canvas_size[1],3),dtype=np.uint8)
    return random_images


if __name__ == '__main__':
    batch_size = 1
    canvas_size = (1000,1000)

    qos = []
    function_total_time = []
    save_time = []
    inference_time = []

    for i in range(10):
        random_images = invoke_detector(batch_size,canvas_size)
        tic = time.perf_counter()
        responses = invoke_keypoint(random_images)
        # print("Function returned: ", responses.text)
        responses = json.loads(responses.text)
        toc = time.perf_counter()
        print(f"invoke_keypoint: {toc - tic:0.4f} seconds")
        qos.append(toc - tic)
        function_total_time.append(responses['total time'])
        save_time.append(responses['save to s3 time'])
        inference_time.append(responses['inference time'])

    print(f"qos: {np.mean(qos):0.4f} seconds")
    print(f"inference_time: {np.mean(inference_time):0.4f} seconds")
    print(f"save_time: {np.mean(save_time):0.4f} seconds")
    print(f"function_total_time: {np.mean(function_total_time):0.4f} seconds")
    