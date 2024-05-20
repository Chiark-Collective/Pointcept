# Use an official NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

ENV PYENV_ROOT=/usr/local/.pyenv \
  PYTHON_VERSION=3.11.8 \
  WORKDIR=/app \
  DEBIAN_FRONTEND=noninteractive \
  TORCH_CUDA_ARCH_LIST=8.9

# Install system packages
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     git \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# install system dependencies (incl build python)
RUN apt update -y \
  && apt -y install netcat gcc curl make openssl systemd git build-essential git \
    curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget \
    xz-utils zlib1g-dev pdal libsparsehash-dev\
  && apt clean

# Set the working directory in the container to /app
WORKDIR $WORKDIR

# -- python
# Set-up necessary Env vars for PyEnv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
# Install pyenv
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash

# Upgrade pip to its latest version
RUN pip install --upgrade pip

# Install PyTorch 2.2.0 with CUDA 12.1 support
# Adjust this line if the exact version for CUDA 12.1 is different or not available.
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install wheel
# Copy the pointops directory contents into the container at /app
COPY ./libs/pointops /app/pointops
COPY ./libs/pointgroup_ops /app/pointgroup_ops

# Set the directory for the pointops to be the working directory
WORKDIR /app

# Run setup.py to build the wheels
RUN cd pointops && TORCH_CUDA_ARCH_LIST=8.9 python setup.py bdist_wheel > /app/build_pointops.log 2>&1 && ls /app/pointops/dist && cd ..
RUN cd pointgroup_ops && TORCH_CUDA_ARCH_LIST=8.9 python setup.py bdist_wheel > /app/build_pointgroup_ops.log 2>&1 && ls /app/pointgroup_ops/dist && cd ..

# copy wheels to wheels directory
RUN mkdir -p /app/wheels
RUN cp /app/pointops/dist/* /app/wheels
RUN cp /app/pointgroup_ops/dist/* /app/wheels
RUN mkdir -p /app/wheels_out

# # Define a command that does nothing; we use the container for building only
CMD ["echo", "Build complete; wheel files are in the mounted volume."]
