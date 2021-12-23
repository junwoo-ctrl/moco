
# default settings.
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
WORKDIR /usr/src/app
SHELL ["/bin/bash", "-c"]


# copy cores.
COPY moco/ ./moco
COPY detection/ ./detection
COPY main_moco.py ./
COPY main_lincls.py ./

# set requirements
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y wget vim git gcc build-essential
RUN apt-get install -y python3 python3-pip
COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

# copy customs.
COPY moco_test/ ./moco_test


# sleep.
CMD ["sleep", "infinity"]
