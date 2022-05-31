# Example: docker build --network=host --build-arg passwd=123 -f Dockerfile -t my_image ..
# Example: docker run -it --rm --gpus all --network host --mount type=bind,source=../../flickr8k,target=/home/user/flickr8k my_image  /bin/bash

# FROM nvidia/cuda:11.0-devel-ubuntu20.04
FROM nvidia/cuda:11.0.3-base-ubuntu20.04 
ENV DEBIAN_FRONTEND noninteractive
ARG passwd
ARG python=python3.10

RUN apt-get update -qq && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y zsh vim git openssh-server tmux screen sudo curl
RUN apt-get update -qq  && apt-get install $python -y
RUN apt-get install -y python3-distutils python3-apt ${python}-distutils 
RUN apt-get install -y ffmpeg libgtk2.0-dev libgl1-mesa-dev perl-modules
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apk/*

ENV NEW_USER user
ENV SHELL /bin/bash
RUN useradd -s $SHELL -m $NEW_USER
RUN gpasswd -a $NEW_USER sudo
RUN echo "${NEW_USER}:${passwd}" | chpasswd
USER $NEW_USER
WORKDIR /home/$NEW_USER

# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | $python
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | $python - -y
RUN echo "PATH=${HOME}/.poetry/bin:$PATH" >> $HOME/.bashrc

COPY --chown=$NEW_USER:$NEW_USER ./ project