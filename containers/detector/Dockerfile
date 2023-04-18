FROM python:3.9-slim-buster
# FROM python:3.9

WORKDIR /app/

COPY . /app/

RUN apt-get update 

RUN apt install -y g++

RUN apt install -y cmake

RUN apt install -y make

RUN apt install -y wget unzip

RUN apt-get -y install build-essential libgtk2.0-dev libgtk-3-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

EXPOSE 5000

CMD ["python", "posenetdocker.py"]