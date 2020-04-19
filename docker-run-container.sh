#!/bin/bash

DATASET_DIR=/datasets
HOME_DIR=/home/allansp
WORK_DIR=/work/allansp


nvidia-docker run --rm -it --ipc=host --userns=host --name allansp-sfsnet-c1 \
    -v ${DATASET_DIR}:${DATASET_DIR} \
    -v ${HOME_DIR}:${HOME_DIR} \
    -v ${WORK_DIR}:${WORK_DIR} \
    allansp-sfsnet:1.0 bash




