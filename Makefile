exec:
	# docker-compose up
	docker-compose run tensorflow bash -c "cd /fsl && rm -r temp && mkdir temp && python main.py"
build:
	docker-compose build
monitor:
	watch -n 0.1 "sensors | grep Core; cat /proc/cpuinfo | grep MHz; nvidia-smi"