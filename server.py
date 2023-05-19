from fastapi import FastAPI, File, UploadFile, Form
from buffer import Image, Table
import cv2
import numpy as np

app = FastAPI()
table1 = Table()
table2 = Table()
switch = False
 
@app.post("/uploadimage/")
async def create_upload_file(file: UploadFile = File(...), created_time: float = Form(...), slo: float = Form(...)):
    global switch,table1,table2
    binary_image = await file.read()
    image_array = cv2.imdecode(np.fromstring(binary_image, np.uint8), cv2.IMREAD_UNCHANGED)
    image = Image(image_array,created_time,slo)
    if switch == False:
        if table1.push(image) == False:
            table2.push(image)
            switch = True
    else:
        if table2.push(image) == False:
            table1.push(image)
            switch = False
    return 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)