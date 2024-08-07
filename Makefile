################################################################################
# Variables
WHEELS_DIR := ./wheels_container

################################################################################
# Targets
.PHONY: all docker-env copy-wheels enter-container local-install ptv3-add ptv3-update test clean copy

# Build the pointcept environment docker image
docker-env:
	docker-compose up -d pointcept-env

enter-container:
	docker exec -it pointcept-env /bin/bash

# Creates a pointcept-builder container that produces wheels for both
# pointops and pointgroup_ops and then copies the wheels to the local host.
# Cleans up container afterwards.
copy-wheels:
	docker-compose up -d pointcept-builder
	mkdir -p wheels_container
	docker cp pointcept-builder:/wheels/. wheels_container
	docker-compose stop pointcept-builder
	docker-compose rm -f pointcept-builder

# Run this for a local install for development. Assumes nothing about your
# system environment, run only if you've prepared your system for Pointcept!
# local-install:
# 	poetry install

################################################################################
# Adds the PTv3 submodule from huggingface.
# Requires ssh-agent on host to run on host or container.
# Does not require running as the git repo already has the submodule initialised.
ptv3-add:
	git lfs install
	mkdir -p models 
	cd models && git submodule add git@hf.co:Pointcept/PointTransformerV3

# Pulls the PTv3 models submodule from huggingface.
# Requires ssh-agent on host to run on host or container.
# Run on host or on container to pull the models.
ptv3-update:
	git lfs install
	mkdir -p models
	cd models && git submodule update --init

################################################################################
# Run the PTv3 test config
test-ptv3:
	poetry run python tools/test.py --config-file $(PTV3_CONFIG_PATH) --options save_path=$(PTV3_SAVE_PATH) weight=$(PTV3_WEIGHTS_PATH)

################################################################################
# Cleans any newly built wheels and de-inits the PTv3 model submodule.
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
