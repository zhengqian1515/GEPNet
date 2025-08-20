import numpy as np
import pickle as pkl
from batchgenerators.utilities.file_and_folder_operations import *

path ='/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET/gepnet_preprocessed/Task503_ecpc/nnFormerPlansv2.1_trgSp_1x1x1_plans_3D.pkl'
with (open(path, 'rb')) as f:
    s = pkl.load(f)
    print(s['plans_per_stage'][0]['batch_size'])
    print(s['plans_per_stage'][0]['patch_size'])
    # print(s)
    #
    # plans = load_pickle(path)
    # plans['plans_per_stage'][0]['batch_size'] = 130
    # plans['plans_per_stage'][0]['patch_size'] = np.array((144, 144))
    #
    # save_pickle(plans, join(r'/home/hpc/xwp/MedNeXt-main/DATASET/nnUNet_raw/nnUNet_cropped_data/Task016_Mcl/nnFormerPlansv2.1_plans_2D.pkl'))
    


if __name__ == '__main__':
    input_file = '/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET/gepnet_preprocessed/Task503_ecpc/nnFormerPlansv2.1_trgSp_1x1x1_plans_3D.pkl'
    output_file = '/data/MedSegmentation/ZXY/GEPNet-main/GepNet_PET/DATASET/gepnet_preprocessed/Task503_ecpc/nnFormerPlansv2.1_trgSp_1x1x1_plans_3D.pkl'
    a = load_pickle(input_file)
    #a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    a['plans_per_stage'][0]['batch_size'] = 4
    save_pickle(a, output_file)