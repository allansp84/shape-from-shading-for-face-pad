#!/bin/bash

device_number=4
project_path=/home/allansp/experimentos/developments/cfpad-env/sfsnet
# base_path=/home/allansp/experimentos/developments/cfpad-env/sfsnet
base_path=/work/allansp
output_path=working


mkdir -p $base_path/$output_path
mkdir -p $project_path/logs


run_feature_extraction()
{
    sfsnet.py --dataset 0 --dataset_path $project_path/dataset/replayattack --face_locations_path $project_path/dataset/replayattack/face-locations --output_path $base_path/$output_path --device_number $device_number --n_jobs 6 --feature_extraction --build_multichannel_input > $project_path/logs/build_multichannel_input.replayattack.log 2>&1
}


run_feature_extraction

