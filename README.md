# DACD-HRTF
Code release for: DACD-HRTF (Core Code comming soon)


### Abstract

Personalized head-related transfer function (HRTF) modeling is essential for immersive spatial audio rendering. However, acquiring high-resolution HRTFs is laborious, as it requires measurements from hundreds or thousands of spatial directions. To alleviate this burden, recent approaches either estimate HRTFs from anatomical data or reconstruct high-resolution HRTFs from a small set of measured directions. Existing methods, however, are often limited to single-dataset training and may fail to generalize to new environments due to dataset-specific measurement biases (e.g., device frequency responses and recording environment setup). In this work, we propose DACD-HRTF, a cross-dataset personalized HRTF upsampling framework based on sparse HRTF measurements. Specifically, DACD-HRTF integrates a direction-aware autoencoder to encode sparse measurements into unified latent representations, which are then decoded to estimate personalized HRTFs over high-resolution spatial directions. This design supports joint training across multiple datasets with heterogeneous spatial sampling grids and measurement conditions. To further mitigate measurement biases across datasets, we introduce a contrastive-based disentanglement strategy that separates subject-specific and dataset-specific latent features, thereby enhancing cross-dataset generalization. Crucially, our framework enables direct reconstruction from sparse measurements without fine-tuning of network parameters at inference, allowing rapid deployment in new environments.
Extensive experiments on eight public HRTF datasets demonstrate that DACD-HRTF achieves competitive performance in reconstructing personalized HRTFs for unseen datasets, showing particular effectiveness under practical sparse measurement settings.

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

- ARI  、BiLi 、CIPIC  、Listen 、HUTUBS 、Sonicom、RIEC 、Crossmod
- **Training/in-dataset evaluation datasets**: ARI, BiLi, CIPIC, Listen, HUTUBS and Sonicom
- **Unseen-dataset evaluation datasets**:RIEC and Crossmod

> **Note**: The released pretrained checkpoint is trained on ARI, BiLi, CIPIC, Listen, HUTUBS and Sonicom.
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
-n ari bili cipic hutubs listen sonicom 
--type_config train
```

- The log is written in outputs/model/logs

- You can see loss graphs in tensorboard by 

  ```bash
  tensorboard --logdir=outputs/model/logs --port=6008
  ```

  
