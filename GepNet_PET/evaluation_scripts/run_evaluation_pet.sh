#!/bin/sh

DATASET_PATH=/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET # dataset path
CHECKPOINT_PATH="$DATASET_PATH"/gepnet_trained_models

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export gepnet_preprocessed="$DATASET_PATH"/gepnet_preprocessed
export gepnet_raw_data_base="$DATASET_PATH"/gepnet_raw


python ../gepnet/run/run_training.py 3d_fullres gepnet_trainer_PET 503 4 -val -p nnFormerPlansv2.1_trgSp_1x1x1
