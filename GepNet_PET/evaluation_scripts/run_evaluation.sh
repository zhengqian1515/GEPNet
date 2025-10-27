#!/bin/sh
export PYTHONPATH=/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET  #
REF_FOLDER="$PYTHONPATH"/DATASET/gepnet_trained_models/gepnet/3d_fullres/Task503_ecpc/gepnet_trainer_PET__nnFormerPlansv2.1_trgSp_1x1x1/gt_niftis   # ground truth
PRED_FOLDER="$PYTHONPATH"/DATASET/gepnet_trained_models/gepnet/3d_fullres/Task503_ecpc/gepnet_trainer_PET__nnFormerPlansv2.1_trgSp_1x1x1/pred_niftis  # pred
LABELS="0 1"  #
CUDA_VISIBLE_DEVICES=9 python ../gepnet/evaluation/evaluator.py -ref $REF_FOLDER -pred $PRED_FOLDER -l $LABELS
