FROM tensorflow/tensorflow:latest-gpu

WORKDIR .
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . ./movenet

CMD ["python", "movenet/movenet.py"]