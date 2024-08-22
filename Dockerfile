# Use an image with GPU support prebuild by Nvidia (https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)
# The --platform parameter is mandatory on ARM Mac to force the build of the container using amd64 (x64). Without this parameter, the container will not work on the CaaS cluster.
# Ubuntu 20.04, NVIDIA CUDA 12.0.1, pytorch 1.14.0a0+44dac51, TensorRT 8.5.31.14.0a0+44dac51
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.02-py3 
# Installing ssh, rsync, rclone, anaconda, vscode-server
# Here Anaconda3-2023.03-1-Linux-x86_64.sh should be downloaded and placed in
# same folder as dockerfile, this image still includes installation of sudo,
# but after applying abovementioned restriction, it will became useless
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
ENV TZ="Europe/Zurich"
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntugis/ppa
RUN apt-get update && apt-get install -y openssh-server sudo rsync rclone git nano screen htop psmisc gdal-bin \
python3-gdal 
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


# Copy your code in the container
#COPY requirements.txt /home/${LDAP_USERNAME}
## Switch to the local user
WORKDIR /home/${LDAP_USERNAME}/
RUN python -m pip install --upgrade pip
RUN python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN python -m pip install transformers
RUN python -m pip install pandas imagecodecs
RUN python -m pip install tqdm wandb yacs argparse matplotlib tifffile
RUN python -m pip install -U scikit-learn
#RUN pip install -r requirements.txt
USER ${LDAP_USERNAME}
# Install additional dependencies