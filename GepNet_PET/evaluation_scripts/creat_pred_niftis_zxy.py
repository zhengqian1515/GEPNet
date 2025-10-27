import os
import shutil


def copy_validation_results(base_folder: str):
    """
    Copies the contents of validation_raw_postprocessed from each fold directory to a new directory pred_niftis.

    :param base_folder: The base folder containing the fold directories.
    """
    folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
    pred_niftis_folder = os.path.join(base_folder, 'pred_niftis')

    os.makedirs(pred_niftis_folder, exist_ok=True)

    for fold in folds:
        validation_folder = os.path.join(base_folder, fold, 'validation_raw_postprocessed')

        if os.path.exists(validation_folder):
            for item in os.listdir(validation_folder):
                if item.endswith('.nii.gz'):
                    src_item = os.path.join(validation_folder, item)
                    dst_item = os.path.join(pred_niftis_folder, item)

                    shutil.copy2(src_item, dst_item)
        else:
            print(f"Warning: {validation_folder} does not exist.")


# 使用示例
base_folder = '/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET/gepnet_trained_models/gepnet/3d_fullres/Task503_ecpc/gepnet_trainer_PET__nnFormerPlansv2.1_trgSp_1x1x1'
copy_validation_results(base_folder)

