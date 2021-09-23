FROM nvcr.io/nvidia/tensorflow:21.06-tf1-py3

RUN \
  pip install -U pip \
  pip install keras==2.2.5 \
  pip install imblearn
# pip install tensorflow-determinism 

WORKDIR /fsl
