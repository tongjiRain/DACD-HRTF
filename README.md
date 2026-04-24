# DACD-HRTF
Code release for: DACD-HRTF (Core Code comming soon)


### Abstract

Personalized head-related transfer function (HRTF) modeling is essential for immersive spatial audio rendering. Conventional personalized methods require laborious and time-consuming measurements, making them impractical for large-scale applications. Recent data-driven methods typically leverage existing measurement datasets, either by upsampling sparse measurements on fixed directions or by directly estimating HRTFs from anatomical geometry information. However, inconsistent spatial sampling directions and anatomical information across datasets limit these methods to single-dataset training and increase the risk of overfitting. In this work, we propose CDP-HRTF, a cross-dataset personalized HRTF estimation framework based on sparse measurements. Specifically, our method design a direction-aware autoencoder architecture that encodes sparsely measured HRTFs into a unified latent representation, which is then decoded to estimate personalized HRTFs at arbitrary spatial directions.  This design enables joint training across datasets with varying spatial sampling directions.  Considering that measured HRTFs are affected by dataset-specific measurement conditions such as equipment and environment, we further develop a contrastive-based disentanglement strategy to explicitly separate dataset-specific features from the unified latent representation.  Extensive experiments on eight public HRTF datasets show that our method outperforms both interpolation-based and neural network baselines in estimating personalized HRTFs for unseen subjects, whether from held-out individuals within the training datasets or from entirely new datasets, using only sparse measurements.

### Requirements

We checked the code with the following computational environment.

* Ubuntu 20.04.2 LTS

* GeForce RTX 3090Ti (24GB VRAM)

* Python 3.9.10

  * ```markdown
    torch==1.13.1
    torchaudio == 0.13.1
    python-sofa == 0.2.0
    teosorboard == 2.19.0
    scipy == 1.13.1
    numpy == 1.26.4
    ```


### Datasets

You can obtain the raw HRTF datasets from the [SOFA conventions website](https://www.sofaconventions.org).

We use the following publicly available datasets in our work:

- ARI  、BiLi 、CIPIC  、Listen 、HUTUBS 、RIEC 、3D3A 、Crossmod 

> **Note**: Two subjects from the ARI dataset (`hrtf_nh10.sofa` and `hrtf_nh22.sofa`) are excluded due to missing measurements in certain directions.

After downloading, the raw data should be organized as follows:

```bash
data/
 ├── ARI/
 │   └── sofa/
 │       └── hrtf_nh*.sofa
 ├── BiLi/
 │   └── sofa/
 │       └── IRC_*_C_HRIR_96000.sofa
 ...
```

Run the preprocessing script located in the `preprocess/` directory:

```bash
run preprocess.py
```

### Project Structure

After dataset collection and preprocessing,  the expected directory layout of the `CDP-HRTF` project is as follows:

```
CDP-HRTF/
 ├── data/                       # Raw SOFA files (downloaded HRTF datasets)
 ├── preprocess/                 # Scripts for converting SOFA → internal format
 │   ├── preprocess.py           # Batch-processing of all SOFA files
 │   └── SOFAdatasets.py         # Dataset loader utilities for SOFA files
 ├── preprocessed_data/          # Full training/evaluation HRTF data (all subjects)
 │   └── HRIR/                   # Pickle files, one per dataset & split
 │       ├── ari_000.pkl
 │       ├── bili_000.pkl
 │       ├── cipic_000.pkl
 │       └── ... # One file per subject per dataset
 ├── preprocessed_data_few/      # Subset for quick checkpoint validation (no download needed)
 │   └── HRIR/
 │       ├── 3d3a_000.pkl
 │       ├── 3d3a_001.pkl
 │       ├── crossmod_000.pkl
 │       ├── crossmod_001.pkl
 │       ├── riec_000.pkl
 │       └── riec_001.pkl
 ├── src/                        # Core source code
 │   ├── configs.py               # Training & evaluation configurations
 │   ├── dataset.py              # Data loading & batching logic
 │   ├── losses.py               # Loss functions (LSD, ILD, contrastive, orthogonality)
 │   ├── model.py                # CDP-HRTF neural network implementation
 │   └── utils.py                # Utility functions (plotting, I/O, etc.)
 ├── t_des/                      # Precomputed t-design sampling grids
 ├── checkpoint/                 # Saved model checkpoints
 │   └── cdp_hrtf.best.net       # Pretrained model on ARI, BiLi, CIPIC, Listen, HUTUBS
 ├── train.py                    # Entry point for training
 └── evaluate_model.py           # Evaluation script for trained models
```

### Quick Start

#### Test

##### 	1. Test Pretrained Model on `preprocessed_data_few`  (no need to collect data)

```bash
python evaluate_model.py 
--dataset_directory ./preprocessed_data_few/HRIR 
--model_file ./checkpoint/cdp_hrtf.best.net 
--type_config test
```

##### 	2. Test  Pretrained Model on `preprocessed_data` (need to collect data and proprocess)

```bash
python evaluate_model.py 
--dataset_directory ./preprocessed_data/HRIR 
--model_file ./checkpoint/cdp_hrtf.best.net 
--type_config test
```

#### Train

##### 	Training the CDP-HRTF with multiple datasets (need to collect data and preprocess)

```bash
python train.py 
--dataset_directory ./preprocessed_data/HRIR 
-n ari bili cipic hutubs listen 
--type_config train
```

- The log is written in outputs/model/logs

- You can see loss graphs in tensorboard by 

  ```bash
  tensorboard --logdir=outputs/model/logs --port=6008
  ```

  
