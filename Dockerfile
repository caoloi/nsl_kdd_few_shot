FROM nvcr.io/nvidia/tensorflow:21.10-tf1-py3

RUN apt update -y && apt install -y fonts-noto-cjk

RUN \
  pip install -U pip \
  # pip install keras==2.2.5 \
  pip install sklearn  \
  pip install pandas \
  pip install matplotlib

WORKDIR /fsl
