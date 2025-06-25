#!/bin/sh

DATASET_PATH=/media/hpc/264CEE814CEE4B5F/ZXY/GEPNet/GepNet_PET/DATASET # dataset path
CHECKPOINT_PATH=/media/hpc/264CEE814CEE4B5F/ZXY/GEPNet/GepNet_PET/DATASET/OUTPUT/test_mcl # checkpoint path

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export gepnet_preprocessed="$DATASET_PATH"/gepnet_preprocessed/Task501_mcl
export gepnet_raw_data_base="$DATASET_PATH"/gepnet_raw/gepnet_raw_data/Task501_mcl

python ../gepnet/run/run_training.py 3d_fullres gepnet_trainer_PET 501 0 -val -p nnUNetPlansv2.1_trgSp_4x4x4
