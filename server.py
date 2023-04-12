import socket
import struct,datetime
import numpy as np
import cv2
from buffer import Buffer

# 创建socket并绑定端口号
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('10.1.81.24', 8888))
server_socket.listen()
print('Server started and listening on port 8888...')
# global count
count = 0
buffer1 = Buffer(10, 480, 640)

while True:
    # 等待客户端连接
    client_socket, client_address = server_socket.accept()
    print('Client connected from', client_address)

    # 接收图像大小
    img_size_data = client_socket.recv(4)
    img_size = struct.unpack('!i', img_size_data)[0]

    # 接收图像数据
    img_data = b''
    while len(img_data) < img_size:
        data = client_socket.recv(img_size - len(img_data))
        if not data:
            break
        img_data += data

    # 将图像数据转换为OpenCV Mat变量
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image_name = datetime.datetime.now().strftime('%m-%d-%H-%M-%S') + str(count) +'.jpg'
    # 保存图像文件
    # cv2.imwrite(image_name, img)
    buffer1.add(img, image_name)
    count += 1
    # 关闭socket
    
client_socket.close()
server_socket.close()
