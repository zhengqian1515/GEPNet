# GEPNet: Granularity-Edge Perception Feature Learning Network for 3D PET Segmentation
## Abstract
> Positron emission tomography (PET) images reveal metabolic activities at the cellular level and play a crucial role in detecting microscopic lesions and assessing prognosis. However, the intra-slice and inter-slice semantic sparsity of 3D PET images poses a great challenge for developing PET segmentation algorithms. To address this problem, we propose a Granularity-Edge Perception Feature Learning Network (GEPNet) for lesion detection and smooth segmentation under semantic sparsity. GEPNet contains Granularity-Aware Downsampling Module (GADM), Aggregation-Driven Lesion Attention (ADLA) module, and Spatial-Channel Adaptive Fusion (SCAF) module. Specifically, to preserve and associate semantic details, GADM retains fine-grained information from original slices by multiple down-sampling branches to minimize semantic loss. The ADLA module introduces aggregation operations within the attention mechanism to effectively capture intra-slice semantic features and establish correlations across inter-slice. Furthermore, the SCAF module adaptively fuses multi-scale local feature maps based on lesion global information to refine lesion detection and capture edge texture features. Finally,  we retrospectively compile the Mantle Cell Lymphoma PET Imaging Diagnosis (MCLID) dataset, comprising 176 patient cases collected from multiple central hospitals. Extensive experiments demonstrate that GEPNet significantly improves segmentation performance on the ECPC-ID, Hecktor 2022, and MCLID datasets. GEPNet excels in detecting small lesions and accurately delineating lesion boundaries, effectively balancing computational efficiency with high segmentation accuracy.

## Updates
- First release – Jun 25, 2024

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

  Please refer to [Setting up the datasets](https://github.com/Amshaker/unetr_plus_plus) on UNETR++ repository for more details. 

## Training
The following scripts can be used for training our GEPNet model on the datasets:
```
bash training_scripts/run_training_pet.sh
```

## Evaluation
For evaluation:
```
bash evaluation_scripts/run_evaluation_pet.sh
```

For inference:
```
nnFormer/nnformer/inference/predict_simple.py
```
## Acknowledgement

## Citation

## Contact
Should you have any question, please create an issue on this repository or contact me at zxy1515pyy@163.com.


