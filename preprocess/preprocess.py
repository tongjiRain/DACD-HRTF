import os
import pickle as pkl
import SOFAdatasets as SOFAdatasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

out_dir = "../preprocessed_data/HRIR"  # HRIRs output directory

def save_hrir_as_pkl():
    if os.path.exists(out_dir):
        raise FileExistsError(f"Directory '{out_dir}' already exists. Please remove it or choose another path.")
    else:
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    dataset_dict = {
        "ari": "ARI",
        "bili": "BiLi",
        "cipic": "CIPIC",
        "listen": "Listen",
        "hutubs": "HUTUBS",
        "sonicom": "SONICOM",
        "riec": "RIEC",
        "3d3a": "Prin3D3A",
        "crossmod": "Crossmod"
    }

    for name in list(dataset_dict.keys()):
        dataset_obj = getattr(SOFAdatasets, dataset_dict[name])()
        for idx in tqdm(range(len(dataset_obj)), desc=f"Processing {name}"):
            locations, hrirs = dataset_obj[idx]
            filename = f"{name}_{idx:03d}.pkl"
            with open(os.path.join(out_dir, filename), 'wb') as handle:
                pkl.dump((locations, hrirs), handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    save_hrir_as_pkl()
