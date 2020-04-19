#!/bin/bash

device_number=4
project_path=/home/allansp/experimentos/developments/cfpad-env/sfsnet
# base_path=/home/allansp/experimentos/developments/cfpad-env/sfsnet
base_path=/work/allansp
output_path=working
ml_algo_options=7
map_type_options="0 1 2 3 4"


mkdir -p $base_path/$output_path
mkdir -p $project_path/logs


run_inter_classification_experiment_1_casia_ra()
{
    for ml_algo in $ml_algo_options
    do
        for map_type in $map_type_options
        do
            sfsnet.py --dataset 0 --dataset_path $project_path/dataset/replayattack --face_locations_path $project_path/dataset/replayattack/face-locations --dataset_b 1 --dataset_path_b $project_path/dataset/cbsr_antispoofing/cbsr.ia.ac.cn --face_locations_path_b $project_path/dataset/cbsr_antispoofing/cbsr.ia.ac.cn/face-locations --output_path $base_path/$output_path --device_number $device_number --map_type $map_type --ml_algo $ml_algo --n_jobs 6 --classification --force_testing > $project_path/logs/inter.classification.casia.ra.$map_type.$ml_algo.log 2>&1

            sfsnet.py --dataset 0 --dataset_path $project_path/dataset/replayattack --face_locations_path $project_path/dataset/replayattack/face-locations --dataset_b 1 --dataset_path_b $project_path/dataset/cbsr_antispoofing/cbsr.ia.ac.cn --face_locations_path_b $project_path/dataset/cbsr_antispoofing/cbsr.ia.ac.cn/face-locations --output_path $base_path/$output_path --device_number $device_number --map_type $map_type --ml_algo $ml_algo --n_jobs 6 --show_results > $project_path/logs/inter.classification.casia.ra.$map_type.$ml_algo.show_results.log 2>&1
        done
    done
}


run_inter_classification_experiment_1_casia_ra

