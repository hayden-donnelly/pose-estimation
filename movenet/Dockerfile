FROM tensorflow/tensorflow:latest-gpu

WORKDIR .
COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install libxcb-xinerama0

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . ./movenet

CMD ["python", "movenet/movenet.py"]