#!/bin/bash

docker build --rm --tag allansp-sfsnet:1.0 \
             --file Dockerfile .


# DATASET_DIR=/datasets
# HOME_DIR=/home/allansp
# WORK_DIR=/work


# nvidia-docker run -it \
#     --ipc=host \
#     --userns=host \
#     --name allansp-cfpad-c1 \
#     -v ${DATASET_DIR}:${DATASET_DIR}/ \
#     -v ${HOME_DIR}:${HOME_DIR}/ \
#     -v ${WORK_DIR}:${WORK_DIR} allansp-cfpad:1.0 \
#     bash




