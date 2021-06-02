FROM nvcr.io/nvidia/tensorflow:21.05-tf1-py3

RUN \
  pip install -U pip \
  pip install keras==2.2.4 \
  pip install imblearn

WORKDIR /fsl
