#!/bin/sh
#
DATASET_PATH=/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET

export PYTHONPATH=.././
export RESULTS_FOLDER="$DATASET_PATH"/gepnet_trained_models
export gepnet_preprocessed="$DATASET_PATH"/gepnet_preprocessed
export gepnet_raw_data_base="$DATASET_PATH"/gepnet_raw

#nnFormer_plan_and_preprocess
python ../gepnet/experiment_planning/nnFormer_plan_and_preprocess.py -t 503 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1

