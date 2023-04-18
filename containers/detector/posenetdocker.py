import time,os,torch,io,json
from minio import Minio
import posenet
import cv2
import numpy as np
from flask import Flask, request
import time, hashlib

model = None
client = None
app = Flask(__name__)
image_name = []
data_storage = "mybucket"

def create_id():
    m = hashlib.md5(str(time.perf_counter()).encode("utf-8"))
    return m.hexdigest()

def init_client():
    global client
    client = Minio(
    # "10.1.81.24:9000", #minio数据库的API地址
    # "10.1.83.80:9000",
    "172.17.0.3:9000",
    # access_key="minioadmin", #RootUser
    access_key="ROOTNAME",
    # secret_key="minioadmin", #RootPassword
    secret_key="CHANGEME123",
    secure=False, #必须加上这一项，否则会按照Amazon S3 Compatible Cloud Storage来处理
)

def load_network():
    global model
    model = posenet.load_model(101)
    
def posedetect(f):
    output_stride = model.output_stride
    nparray = f
    tensor = torch.from_numpy(f).to(torch.float32).transpose(1,3).transpose(2,3)
    draw_image_pool = []
    output_scale_pool = []
    for i in nparray:
        _, draw_image, output_scale = posenet.read_imgfile(i, scale_factor=1.0, output_stride=output_stride)
        draw_image_pool.append(draw_image)
        output_scale_pool.append(output_scale)
    with torch.no_grad():
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(tensor)
        cunt = 0
        for heatmap, offset, displacement_fwd, displacement_bwd, draw_image in zip(heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result, draw_image_pool):
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmap,
                offset,
                displacement_fwd,
                displacement_bwd,
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale[cunt]

            draw_image = posenet.draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords,min_pose_score=0.25, min_part_score=0.25)
            put_in_bucket(draw_image)
            cunt += 1

    return cunt

def put_in_bucket(image,data_storage = data_storage):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    name = str(create_id()) + ".jpg"
    image_name.append(name)
    client.put_object(data_storage, name, io_buf, len(io_buf.getvalue()),content_type="image/jpg")
    return

def handle(np_data):
    try:
        if model is None:
            load_network()
        if client is None:
            init_client()
        start = time.time()         
        batch_size = posedetect(np_data) #处理图片
        response_body = {
            'Batch Size' : batch_size,
            'Image Names': image_name,
            'Data storage': data_storage,
            'Elapse time': time.time() - start,
        }
        image_name = []
        return json.dumps(response_body)
    except Exception as e:
        print(str(e))
        return json.dumps({'error': 'Internal Server Error','message': str(e)})


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def main_route(path):
    if (request.method == 'POST'):
        payload_str = request.data.decode('utf-8')
        payload = json.loads(payload_str)
        shape = tuple(payload['shape'])
        batch_data = np.array(payload['data'],dtype=np.uint8).reshape(shape)
        ret = handle(batch_data)
        return ret
    else:
        return request.method

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)