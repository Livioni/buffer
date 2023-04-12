import requests,json,io
from PIL import Image

functions = {
    'debug' : 'http://10.1.81.24:31112/async-function/debug',
    'keypoint' : 'http://10.1.81.24:31112/async-function/keypoint'
}
def ndarray2bytes(img_arr):
    """ndarray的图片转换成bytes"""
    imgByteArr = io.BytesIO()
    Image.fromarray(img_arr).save(imgByteArr, format='jpeg')
    img_data = imgByteArr.getvalue()
    return img_data

def invoke_debug(input_data):
    '''
    endpoint: the name of the function
        'debug';
        'keypoint'
    input_data: the image data
    '''
    endpoint_url = functions['debug']
    # Convert the image to hex format
    img_data = ndarray2bytes(input_data)
    # Send the HTTP POST request
    response = requests.post(endpoint_url, files={'file': img_data},timeout=30)
    return response

def invoke_keypoint(input_data):
    '''
    endpoint: the name of the function
        'debug';
        'keypoint'
    input_data: the image data
    '''
    endpoint_url = functions['keypoint']
    # Convert the image to hex format
    hex_img = input_data.hex()
    # Construct the input JSON payload
    input_data = {'image': hex_img}
    send_data = json.dumps(input_data)
    # Send the HTTP POST request
    response = requests.post(endpoint_url, data=send_data,timeout=30)
    return response