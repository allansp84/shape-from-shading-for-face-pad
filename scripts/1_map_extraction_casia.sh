#!/bin/bash

device_number=4
project_path=/home/allansp/experimentos/developments/cfpad-env/sfsnet
# base_path=/home/allansp/experimentos/developments/cfpad-env/sfsnet
base_path=/work/allansp
output_path=working


mkdir -p $base_path/$output_path
mkdir -p $project_path/logs


run_map_extraction()
{
    sfsnet.py --dataset 1 --dataset_path $project_path/dataset/cbsr_antispoofing/cbsr.ia.ac.cn --face_locations_path $project_path/dataset/cbsr_antispoofing/cbsr.ia.ac.cn/face-locations --output_path $base_path/$output_path --device_number $device_number --n_jobs 6 --map_extraction > $project_path/logs/map_extraction.casia.log 2>&1
}


run_map_extraction

