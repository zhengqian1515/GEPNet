#!/bin/sh
#
DATASET_PATH=/media/hpc/264CEE814CEE4B5F/ZXY/GEPNet/GepNet_PET/DATASET

export PYTHONPATH=.././
export RESULTS_FOLDER="$DATASET_PATH"/gepnet_trained_models
export gepnet_preprocessed="$DATASET_PATH"/gepnet_preprocessed
export gepnet_raw_data_base="$DATASET_PATH"/gepnet_raw

#pet
#python ../gepnet/run/run_training.py 3d_fullres gepnet_trainer_PET 501 0 -p nnUNetPlansv2.1_trgSp_4x4x4
python ../gepnet/run/run_training.py 3d_fullres gepnet_trainer_PET 1 0 -p nnUNetPlansv2.1
