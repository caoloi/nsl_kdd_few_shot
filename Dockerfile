FROM nvcr.io/nvidia/tensorflow:21.09-tf1-py3

RUN \
  pip install -U pip \
  # pip install keras==2.2.5 \
  pip install sklearn  \
  pip install pandas \
  pip install matplotlib

WORKDIR /fsl
