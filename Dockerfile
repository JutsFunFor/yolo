# Start FROM Ubuntu image https://hub.docker.com/_/ubuntu
FROM ubuntu:22.10

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' package
RUN apt update -y && sudo apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    sudo wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && \
    sudo tar xzf Python-3.8.10.tgz && \
    cd Python-3.8.10 && \
    sudo ./configure --enable-optimizations && \
    sudo make altinstall


RUN apt update \
    && apt install --no-install-recommends -y python3-pip git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++
# RUN alias python=python3

# Create working directory
RUN mkdir -p /usr/src/ultralytics
WORKDIR /usr/src/ultralytics

# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)
RUN git clone https://github.com/ultralytics/ultralytics /usr/src/ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt /usr/src/ultralytics/

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -e . thop --extra-index-url https://download.pytorch.org/whl/cpu

COPY setup_libs.sh ./
RUN sh setup_libs.sh

COPY requirements.txt ./
RUN pip install asyncio-nats-streaming

EXPOSE 554
EXPOSE 555
EXPOSE 556
EXPOSE 557

RUN apt-get install libopenblas-base libopenmpi-dev libomp-dev -y
RUN pip install -r requirements.txt

CMD ["python3"]