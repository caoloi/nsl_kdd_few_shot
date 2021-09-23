exec:
	# docker-compose up
	docker-compose run --rm tensorflow bash -c "cd /fsl && rm -r temp && mkdir temp && python main.py"
	# docker-compose run --rm tensorflow bash -c "cd /fsl && python main.py"
build:
	docker-compose build
monitor:
	watch -n 1 "sensors | grep Core; cat /proc/cpuinfo | grep MHz; nvidia-smi"