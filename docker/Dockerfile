# Docker config file for building image for Group-Diffusion
# This config file is based on that provided in (https://github.com/openai/consistency_models)
#
# Date  : 2023-12-04
#
# Note, setup.py is not longer supported by setuptools > v58.2
# If you encounter any issues relating to package installations please verify the version of setuptools.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libopenmpi-dev git wget \
    python3 python3-dev python3-pip python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc\
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc

RUN pip3 install --no-cache-dir --upgrade pip setuptools==58.2.0 wheel packaging mpi4py 
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 
RUN pip3 install flash-attn==0.2.8
RUN pip3 install blobfile scipy piq pillow tqdm 

WORKDIR /home/
# Install project github dependencies
RUN git clone https://github.com/SpencerSzabados/GrouPy.git 
WORKDIR /home/GrouPy
RUN python3 setup.py install 

# Install project 
RUN git clone https://github.com/SpencerSzabados/Group-Diffusion.git
WORKDIR /home/Group-Diffusion

RUN ln -s /usr/bin/python3 /usr/bin/python
