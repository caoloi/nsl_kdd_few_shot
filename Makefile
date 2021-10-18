exec:
	# docker-compose up
	docker-compose run --rm tensorflow bash -c "cd /fsl && rm -r temp && mkdir temp && python main.py"
	# docker-compose run --rm tensorflow bash -c "cd /fsl && python main.py"
build:
	docker-compose build
create_benchmark:
	docker-compose run --rm tensorflow bash -c "cd /fsl && python create_benchmark_dataset.py"
monitor:
	# watch -n 1 "sensors \
	# 	&& (cat /proc/cpuinfo | grep MHz) \
	# 	&& nvidia-smi \
	# 	&& (sudo nvme smart-log /dev/nvme0n1 | grep temperature) \
	# 	&& (sudo nvme smart-log /dev/nvme1n1 | grep temperature)"
	watch -n 1 "sensors \
		&& nvidia-smi \
		&& (sudo nvme smart-log /dev/nvme0n1 | grep temperature) \
		&& (sudo nvme smart-log /dev/nvme1n1 | grep temperature)"