FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04 

LABEL maintainer "ykic.p3@gmail.com"

ARG UID
ARG GID

# install apt packages
RUN apt-get update && \
  apt-get -y install \
  gcc sudo language-pack-ja xz-utils file curl \
  libhdf5-serial-dev software-properties-common

# install python3.7
ARG PYTHON=python3.7
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y ${PYTHON}
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | ${PYTHON}
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# locale setting
RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# add docker user
RUN groupadd -g $GID docker
RUN useradd -m --uid $UID --gid $GID --shell /bin/bash docker

# WORKDIR
WORKDIR /home/docker
ENV HOME /home/docker

# install python packages
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml /home/docker
COPY poetry.lock /home/docker
RUN poetry install --no-interaction --no-dev

EXPOSE 8889
