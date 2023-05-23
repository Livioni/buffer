from fastapi import FastAPI, File, UploadFile, Form
from buffer import Image, Table
import cv2,time
import numpy as np

app = FastAPI()
table1 = Table(1000,1000,0.2)
time.sleep(1)
table2 = Table(1000,1000,0.2)
switch = 0
 
@app.post("/uploadimage/")
async def create_upload_file(file: UploadFile = File(...), created_time: float = Form(...), slo: float = Form(...)):
    global switch,table1,table2
    binary_image = await file.read()
    image_array = cv2.imdecode(np.frombuffer(binary_image, np.uint8), cv2.IMREAD_UNCHANGED)
    image = Image(image_array,created_time,slo)
    if switch == 0:
        if table1.push(image) == False:
            table2.push(image)
            switch = 1
    elif switch == 1:
        if table2.push(image) == False:
            table1.push(image)
            switch = 0    
    return 

@app.get("/gettable/")
async def get_table():
    global table1,table2
    table1.show_info()
    table2.show_info()
    return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)