FROM tensorflow/tensorflow:latest-gpu

WORKDIR /project
COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install libxcb-xinerama0

RUN pip install --upgrade pip
RUN pip install -U -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES=all

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser" , "--NotebookApp.token=''","--NotebookApp.password=''"]