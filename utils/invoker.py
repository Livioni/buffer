import requests,json,io
from PIL import Image

functions = {
    'keypoint' : 'http://10.1.81.24:5000',
}

def ndarray2bytes(img_arr):
    """ndarray的图片转换成bytes"""
    imgByteArr = io.BytesIO()
    Image.fromarray(img_arr).save(imgByteArr, format='jpeg')
    img_data = imgByteArr.getvalue()
    return img_data


def invoke_keypoint(input_data):
    '''
    endpoint: the name of the function
        'debug';
        'keypoint'
    input_data: the image data
    '''
    endpoint_url = functions['keypoint']
    img_data = ndarray2bytes(input_data)    
    # Send the HTTP POST request
    response = requests.post(endpoint_url, files={'file': img_data},timeout=30)
    return response

