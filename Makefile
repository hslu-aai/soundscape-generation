.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := soundscape-generation
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
# docker
DOCKERCMD := docker run -v $$PWD:/tf/ -it $(PROJECTNAME):$(GIT_BRANCH)
RUNCMD := BUILD_ENV=$(ENV) docker-compose up --build -d

###########################
# COMMANDS
###########################
# Thanks to: https://stackoverflow.com/a/10858332
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

###########################
# PROJECT UTILS
###########################
.PHONY: init
init:  ##@Utils initializes the project and pulls all the nessecary data
	@git submodule update --init --recursive

.PHONY: update_data_ref
update_data_ref:  ##@Utils updates the reference to the submodule to its latest commit
	@git submodule update --remote --merge

.PHONY: clean
clean:  ##@Utils cleanes the project
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f .coverage

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker runs an interacitve bash inside the docker image
	@echo "Run inside docker image"
	$(DOCKERCMD) /bin/bash

###########################
# COMMANDS
###########################
.PHONY: predict_object_detection
predict_object_detection:  ##@Commands predicts the segmentation mask for a folder of images
	@docker-compose up predict_object_detection

.PHONY: sound_generation
sound_generation:  ##@Commands generates a sound file for each segmented image of a folder
	@docker-compose up sound_generation
