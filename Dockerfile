# FROM ubuntu:22.04 AS builder-image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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

# Install ASAP

RUN : \
    && apt-get update \
    && apt-get -y install curl git \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb" \
    && dpkg --install ASAP-2.2-Ubuntu2204.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.11/site-packages/asap.pth \
    && rm ASAP-2.2-Ubuntu2204.deb \
    && :

# install TIAToolbox and its requirements
RUN apt-get update && apt-get install --no-install-recommends -y \
    libopenjp2-7-dev libopenjp2-tools \
    openslide-tools \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tiatoolbox

# install segmentation-models-pytorch
RUN pip install --no-cache-dir segmentation-models-pytorch

# install wholeslidedata
RUN pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main

# activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# folders and permissions
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /tempoutput /input /output /output/images /output/images/breast-cancer-segmentation-for-tils
RUN chown -R algorithm:algorithm /opt/algorithm /input /output /tempoutput

USER algorithm
WORKDIR /opt/algorithm

# RUN git clone -b docker https://github.com/Jiaqi-Lv/TIAger-Torch.git
# RUN chmod -R 755 TIAger-Torch/
COPY --chown=algorithm:algorithm ./ /opt/algorithm/TIAger-Torch
ADD --chown=algorithm:algorithm ./testinput /input/

# Compute requirements
LABEL processor.cpus="1"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="32G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="12G"

WORKDIR /opt/algorithm/TIAger-Torch
# ENTRYPOINT python l1_pipeline.py
RUN ["chmod", "+x", "./commands.sh"]
ENTRYPOINT ["./commands.sh"]