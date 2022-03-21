From nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /root/projects

COPY Arial.ttf /root/.config/Ultralytics/Arial.ttf

COPY requirements.txt /root/projects/requirements.txt

RUN apt-get update \
    && apt-get -y install vim libgl1-mesa-glx

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
