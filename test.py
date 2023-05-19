import requests,os,time,json

files_path = 'test/images'
files = os.listdir(files_path)
end_point = 'http://127.0.0.1:8002/uploadimage/'
for file in files:
    if file == '.DS_Store':
        continue
    file = open(os.path.join(files_path,file), 'rb')
    SLO = 1.0
    file_name = 'test.jpg'
    multipart_form_data = {
        'file': (file),
        'created_time': (None, str(time.time())),
        'slo': (None, str(SLO)),
    }
    response = requests.post(end_point, files=multipart_form_data)
    print(response.json())
    break