exec:
	docker run -m 32g --cpus 16 --gpus all -it --rm -v /home/tksfjt/nsl_kdd_few_shot:/workspace/fsl nvcr.io/nvidia/tensorflow:20.11-tf1-py3 bash -c 'pip install keras==2.2.4 imblearn && cd fsl && python main.py'
