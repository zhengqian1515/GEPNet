#!/bin/sh
#
DATASET_PATH=/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET

export PYTHONPATH=.././
export RESULTS_FOLDER="$DATASET_PATH"/gepnet_trained_models
export gepnet_preprocessed="$DATASET_PATH"/gepnet_preprocessed
export gepnet_raw_data_base="$DATASET_PATH"/gepnet_raw

#pet
CUDA_VISIBLE_DEVICES=9 python ../gepnet/run/run_training.py 3d_fullres gepnet_trainer_PET 503 2 -p nnFormerPlansv2.1_trgSp_1x1x1

