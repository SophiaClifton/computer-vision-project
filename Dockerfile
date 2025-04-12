FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /app

# build opencv with CUDA support
COPY ./docker/build_opencv.sh .
RUN bash ./build_opencv.sh

COPY ./requirements.txt .
RUN pip install -r requirements.txt

CMD python demo/app.py
