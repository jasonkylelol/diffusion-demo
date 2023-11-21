#!/bin/bash

docker run -it --rm --cpus 8 -m 16G --gpus all --name diffusion-demo -p 37860:7860 \
	-v /home/huggingface/tutorials:/app/tutorials \
	-v /home/huggingface/models:/app/models \
	develop-env:v1.0 bash

