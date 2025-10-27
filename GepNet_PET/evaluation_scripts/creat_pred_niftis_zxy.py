import os
import shutil


def copy_validation_results(base_folder: str):
    """
    Copies the contents of validation_raw_postprocessed from each fold directory to a new directory pred_niftis.

    :param base_folder: The base folder containing the fold directories.
    """
    # 定义源文件夹和目标文件夹
    folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
    pred_niftis_folder = os.path.join(base_folder, 'pred_niftis')

    # 创建新文件夹 pred_niftis
    os.makedirs(pred_niftis_folder, exist_ok=True)

    # 遍历每个 fold 目录
    for fold in folds:
        validation_folder = os.path.join(base_folder, fold, 'validation_raw_postprocessed')

        # 检查 validation_raw_postprocessed 文件夹是否存在
        if os.path.exists(validation_folder):
            # 复制以 .nii.gz 结尾的文件到新的 pred_niftis 文件夹
            for item in os.listdir(validation_folder):
                if item.endswith('.nii.gz'):
                    src_item = os.path.join(validation_folder, item)
                    dst_item = os.path.join(pred_niftis_folder, item)

                    # 复制文件，保持元数据
                    shutil.copy2(src_item, dst_item)
        else:
            print(f"Warning: {validation_folder} does not exist.")


# 使用示例
base_folder = '/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET/gepnet_trained_models/gepnet/3d_fullres/Task503_ecpc/gepnet_trainer_PET__nnFormerPlansv2.1_trgSp_1x1x1'
copy_validation_results(base_folder)
