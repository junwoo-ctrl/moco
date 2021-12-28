
# default settings.
#FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM determinedai/environments:cuda-11.2-pytorch-1.7-lightning-1.2-tf-2.5-gpu-84e8332
WORKDIR /usr/src/app
SHELL ["/bin/bash", "-c"]


# copy cores.
COPY moco/ ./moco
COPY detection/ ./detection
COPY main_moco.py ./
COPY main_lincls.py ./

# cuda settings.
#RUN  apt-get update \
#  && apt-get install -y wget
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
#RUN apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
#RUN apt-get update
#RUN apt-get -y install cuda

# set requirements
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y wget vim git gcc build-essential
RUN apt-get install -y python3 python3-pip
COPY requirements.txt ./
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
RUN python3 -m pip install -r requirements.txt

# copy customs.
COPY moco_test/ ./moco_test


# sleep.
CMD ["sleep", "infinity"]
