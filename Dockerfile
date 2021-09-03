# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

LABEL maintainer='Microsoft Alvin Zheng<Ningxin.Zheng@microsoft.com>'

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y install sudo \
    apt-utils \
    git \
    curl \
    vim \
    unzip \
    wget \
    make \
    build-essential \
    cmake \
    libopenblas-dev \
    automake \
    openssh-client \
    openssh-server \
    lsof \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    libcupti-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#
# update pip
#
RUN python3 -m pip install --upgrade pip setuptools==39.1.0

# numpy 1.14.3  scipy 1.1.0
RUN python3 -m pip --no-cache-dir install \
    numpy==1.14.3 scipy==1.1.0

#
# Tensorflow 1.10.0
#
# RUN python3 -m pip --no-cache-dir install tensorflow-gpu==1.10.0

#
# Keras 2.1.6
#
# RUN python3 -m pip --no-cache-dir install Keras==2.1.6

#
# PyTorch
#
RUN python3 -m pip --no-cache-dir install torch==1.9.0
RUN python3 -m pip install torchvision

#
# sklearn 0.20.0
#
RUN python3 -m pip --no-cache-dir install scikit-learn==0.20.0

#
# pandas==0.23.4 lightgbm==2.2.2
#
RUN python3 -m pip --no-cache-dir install pandas==0.23.4 lightgbm==2.2.2

#
# Install NNI
#
RUN git clone  https://github.com/zheng-ningxin/nni.git
# RUN python3 -m pip --no-cache-dir install nni

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbin

WORKDIR /root
