# GEPNet: Granularity-Edge Perception Feature Learning Network for 3D PET Segmentation
## Abstract
> Positron emission tomography (PET) images reveal metabolic activities at the cellular level and play a crucial role in detecting microscopic lesions and assessing prognosis. However, the intra-slice and inter-slice semantic sparsity of 3D PET images poses a great challenge for developing PET segmentation algorithms. To address this problem, we propose a Granularity-Edge Perception Feature Learning Network (GEPNet) for lesion detection and smooth segmentation under semantic sparsity. GEPNet contains Granularity-Aware Downsampling Module (GADM), Aggregation-Driven Lesion Attention (ADLA) module, and Spatial-Channel Adaptive Fusion (SCAF) module. Specifically, to preserve and associate semantic details, GADM retains fine-grained information from original slices by multiple down-sampling branches to minimize semantic loss. The ADLA module introduces aggregation operations within the attention mechanism to effectively capture intra-slice semantic features and establish correlations across inter-slice. Furthermore, the SCAF module adaptively fuses multi-scale local feature maps based on lesion global information to refine lesion detection and capture edge texture features. Finally,  we retrospectively compile the Mantle Cell Lymphoma PET Imaging Diagnosis (MCLID) dataset, comprising 176 patient cases collected from multiple central hospitals. Extensive experiments demonstrate that GEPNet significantly improves segmentation performance on the ECPC-ID, Hecktor 2022, and MCLID datasets. GEPNet excels in detecting small lesions and accurately delineating lesion boundaries, effectively balancing computational efficiency with high segmentation accuracy.

Click [here](GEPNet_User_Guide.ipynb) to view the GEPNet user guide.

## Updates
- update code and readme - August 20, 2025
- First release – Jun 25, 2025

## Installation
The code is tested with Python 3.9, PyTorch 2.0.1 and CUDA 11.7. After cloning the repository, follow the below steps for installation,
  1. Create and activate conda environment
  ```
  conda create --name gepnet python=3.9
  conda activate gepnet
  ```
  2. Install PyTorch and torchvision
  ```
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8  -c nvidia
  ```
  3. Install other dependencies
  ```
  pip install -r requirements.txt
  ```
  4. Install DCN (Deformable Convolutional Networks) repository
  ```
  cd ../dcn
  bash make.sh
  ```

## Dataset
We follow the same dataset preprocessing as in [UNETR++](https://github.com/Amshaker/unetr_plus_plus). We conducted extensive experiments on three benchmarks: MCLID, ECPC-ID, and Hecktor 2022.

1. Dataset download
   
  
  Datasets can be acquired via following links:

  Dataset I MCLID
  
  Dataset II [ECPC-IDS](https://figshare.com/articles/dataset/ECPC-IDS/23808258) 
  
  Dataset III [Hecktor 2022](https://hecktor.grand-challenge.org/) 

2. Setting up the datasets
  After you have downloaded the datasets, you can follow the settings in [UNETR++](https://github.com/Amshaker/unetr_plus_plus) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:
  ```
  ./gepnet/
  ./DATASET/
    ├── gepnet_raw/
        ├── gepnet_raw_data/
            ├── Task501_mcl/
                ├── imagesTr/
                ├── imagesTs/
                ├── labelsTr/
                ├── labelsTs/
                ├── dataset.json
            ├── Task502_ecpc/
                ├── imagesTr/
                ├── imagesTs/
                ├── labelsTr/
                ├── labelsTs/
                ├── dataset.json
            ├── Task503_hecktor/
                ├── imagesTr/
                ├── imagesTs/
                ├── labelsTr/
                ├── labelsTs/
                ├── dataset.json
        ├── gepnet_cropped_data/
    ├── gepnet_trained_models/
    ├── gepnet_preprocessed/
  ```
  After that, you can preprocess the above data using following commands:
  ```
  nnFormer_plan_and_preprocess -t XXX
  ```

  Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details. 

## Training
The following scripts can be used for training our GEPNet model on the datasets:
```
bash training_scripts/run_training_pet.sh
```

## Evaluation
<table>
  <tr>
    <th>Dataset</th>
    <th colspan="5" style="text-align: center">Pre-Trained Weights</th>
  </tr>
  <tr>
    <td>ECPC-IDS</td>
    <td><a href="https://github.com/zhengqian1515/weight_repo/raw/main/fold_0/model_final_checkpoint.model?download=">fold 0</a></td>
    <td><a href="https://github.com/zhengqian1515/weight_repo/raw/main/fold_1/model_final_checkpoint.model?download=">fold 1</a></td>
    <td><a href="https://github.com/zhengqian1515/weight_repo/raw/main/fold_2/model_final_checkpoint.model?download=">fold 2</a></td>
    <td><a href="https://github.com/zhengqian1515/weight_repo/raw/main/fold_3/model_final_checkpoint.model?download=">fold 3</a></td>
    <td><a href=https://github.com/zhengqian1515/weight_repo/raw/main/fold_4/model_final_checkpoint.model?download=#">fold 4</a></td>
  </tr>
</table>

1- Download ECPC-IDS weights and paste model_final_checkpoint.model in the following path, using fold 0 as an example:

```
GEPNet-main/GepNet_PET/DATASET/gepnet_trained_models/gepnet/3d_fullres/Task503_ecpc/gepnet_trainer_PET__nnFormerPlansv2.1_trgSp_1x1x1/fold_0
```

2- Then, run
```
bash evaluation_scripts/run_evaluation_pet.sh
```

3- Finally, take the average of the five cross-validation results.

## Acknowledgement
This repository is built based on [nnFormer](https://github.com/282857341/nnFormer), [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [UNETR++](https://github.com/Amshaker/unetr_plus_plus), [LHUnet](https://github.com/xmindflow/LHUNet). We thank the authors for their code repositories.

## Citation

## Contact
Should you have any question, please create an issue on this repository or contact me at zxy1515pyy@163.com.


