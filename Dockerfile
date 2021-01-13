FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3

RUN pip install -U pip

RUN pip install keras==2.2.4

RUN pip install imblearn

RUN mkdir temp

# COPY exec.sh exec.sh
# COPY *.py *.py

# ENTRYPOINT sh -x exec.sh