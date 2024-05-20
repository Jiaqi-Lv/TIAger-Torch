FROM ubuntu:22.04 AS builder-image

# To avoid tzdata blocking the build with frontend questions
ENV DEBIAN_FRONTEND=noninteractive

# Install python3.11
RUN apt-get update && \
    apt install software-properties-common -y &&\
    add-apt-repository ppa:deadsnakes/ppa -y && apt update &&\
    apt-get install -y --no-install-recommends python3.11-venv &&\
    apt-get install libpython3.11-de -y &&\
    apt-get install python3.11-dev -y &&\
    apt-get install build-essential -y &&\
    apt-get clean

# Install git
RUN apt-get update && apt-get install -y git

# Add env to PATH
RUN python3.11 -m venv /venv
ENV PATH=/venv/bin:$PATH

# install TIAToolbox and its requirements
RUN apt-get update && apt-get install --no-install-recommends -y \
    libopenjp2-7-dev libopenjp2-tools \
    openslide-tools \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tiatoolbox

# install segmentation-models-pytorch
RUN pip install --no-cache-dir segmentation-models-pytorch

# activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# folders and permissions
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output
RUN chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm

RUN git clone -b docker https://github.com/Jiaqi-Lv/TIAger-Torch.git
RUN chmod -R 755 TIAger-Torch/