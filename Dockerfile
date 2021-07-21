FROM tensorflow/tensorflow:2.4.0-gpu

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y libtcmalloc-minimal4
RUN apt-get install -y sox

RUN apt install -y libsndfile1
RUN apt install -y libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip

WORKDIR /tf

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir

COPY . /tf/

RUN ./scripts/install.sh
