# Variables
POINTOPS_BUILDER_IMAGE := pointops-builder
WHEELS_DIR := ./wheels
POINTOPS_WHEEL := $(WHEELS_DIR)/pointops-1.0-cp311-cp311-linux_x86_64.whl
POINTGROUP_OPS_WHEEL := $(WHEELS_DIR)/pointgroup_ops-0.0.0-cp311-cp311-linux_x86_64.whl

# Targets
.PHONY: all install data pointtransformerv3 test clean

all: install data pointtransformerv3

# build wheels and install python environment
install: wheels
	poetry lock --no-update
	poetry install
	poetry run pip install flash_attn --no-build-isolation

# build wheels for pointops and pointgroup_ops in docker container and deposit the .whl files in ./wheels
wheels: $(POINTOPS_WHEEL) $(POINTGROUP_OPS_WHEEL)

$(POINTOPS_WHEEL) $(POINTGROUP_OPS_WHEEL):
	docker build -t $(POINTOPS_BUILDER_IMAGE) .
	mkdir -p $(WHEELS_DIR)
	docker run --gpus all -v $(WHEELS_DIR):/app/wheels_out -it $(POINTOPS_BUILDER_IMAGE) /bin/bash -c "cp /app/wheels/* /app/wheels_out/"

# add the point transformer v3 repo as a submodule
ptv3:
	git lfs install
	mkdir -p models 
	cd models && git submodule add git@hf.co:Pointcept/PointTransformerV3

# run the PTv3 test config
test-ptv3:
	poetry run python tools/test.py --config-file $(PTV3_CONFIG_PATH) --options save_path=$(PTV3_SAVE_PATH) weight=$(PTV3_WEIGHTS_PATH)

clean:
	@if [ -d "$(wildcard $(WHEELS_DIR))" ]; then \
			rm -rf "$(WHEELS_DIR)"; \
	fi
	@if [ -d "./models/PointTransformerV3" ]; then \
		git submodule deinit ./models/PointTransformerV3; \
		rm -rf .git/modules/PointTransformerV3; \
		git rm -f ./models/PointTransformerV3; \
		sed -i '/\[submodule "PointTransformerV3"\]/,+2d' .gitmodules; \
	fi
