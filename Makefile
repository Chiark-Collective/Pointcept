# Variables
POINTOPS_BUILDER_IMAGE := pointops-builder
WHEELS_DIR := ./wheels
S3DIS_DATA_URL := https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/ERtd0QAyLGNMs6vsM4XnebcBseQ8YTL0UTrMmp11PmQF3g?e=MsER95
SCANNET_DATA_URL := https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EREuB1If2DNEjz43-rdaVf4B5toMaIViXv8gEbxr9ydeYA?e=ffXeG4

# Targets
.PHONY: all install data pointtransformerv3 test clean

all: install data pointtransformerv3

# build wheels and install python environment
install: wheels
	poetry install
	poetry run pip install flash_attn --no-build-isolation

# build wheels for pointops and pointgroup_ops in docker container and deposit the .whl files in ./wheels
wheels:
	docker build -t $(POINTOPS_BUILDER_IMAGE) .
	mkdir -p $(WHEELS_DIR)
	docker run --gpus all -v $(WHEELS_DIR):/app/wheels_out -it $(POINTOPS_BUILDER_IMAGE) /bin/bash -c "cp /app/wheels/* /app/wheels_out/"

# WIP: to automate downloading dataset
data:
	# Download and extract S3DIS dataset
	wget -O ./data/s3dis.zip $(S3DIS_DATA_URL)
	unzip ./data/s3dis.zip -d ./data/s3dis
	rm ./data/s3dis.zip

	# Download and extract ScanNet dataset
	wget -O ./data/scannet.zip $(SCANNET_DATA_URL)
	unzip ./data/scannet.zip -d ./data/scannet
	rm ./data/scannet.zip

# add the point transformer v3 repo as a submodule
ptv3:
	git lfs install
	mkdir -p models && cd models
	git submodule add git@hf.co:Pointcept/PointTransformerV3

# run the PTv3 test config
test-ptv3:
	poetry run python tools/test.py --config-file $(PTV3_CONFIG_PATH) --options save_path=$(PTV3_SAVE_PATH) weight=$(PTV3_WEIGHTS_PATH)

clean:
	rm -rf $(WHEELS_DIR)
	rm -rf ./data/s3dis
	rm -rf ./data/scannet
	rm -rf ./models/PointTransformerV3
