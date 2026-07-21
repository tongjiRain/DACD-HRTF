import os, glob
import random
import torch
import torch as th
import torch.nn.functional as F
import torchaudio as ta
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset

class HRTFDataset(Dataset):
    def __init__(self, hrir_path="../preprocessed_data/HRIR", dataset="ari", norm_way=0,
                 sampling_rate=44100, max_f=22050):
        """
        Args:
            hrir_path: path to preprocessed HRIR files (.pkl)
            dataset: name of the dataset (e.g., 'ari', 'cipic')
            norm_way: normalization strategy (0 = none, 1 = max, 2 = top-5% energy, 3 = equator energy)
        """
        self.hrir_path = hrir_path
        self.name = dataset
        self.dataset_len = len(glob.glob(os.path.join(hrir_path, f"{self.name}*.pkl")))
        self.norm_way = norm_way
        self.filter_length = 256 // 2      # number of positive FFT bins (128)
        self.sampling_rate = sampling_rate
        self.max_f = max_f                 # Nyquist frequency for 44.1 kHz HRIRs
        self.HRTFs, self.locations = self._get_allhrtf(self.norm_way)

        # Split into train (80%), validation (10%), test (10%)
        self.HRTFs_train = self.HRTFs[:int(0.8 * self.dataset_len)]
        self.locations_train = self.locations[:int(0.8 * self.dataset_len)]

        self.HRTFs_valid = self.HRTFs[int(0.8 * self.dataset_len):int(0.9 * self.dataset_len)]
        self.locations_valid = self.locations[int(0.8 * self.dataset_len):int(0.9 * self.dataset_len)]

        self.HRTFs_test = self.HRTFs[int(0.9 * self.dataset_len):]
        self.locations_test = self.locations[int(0.9 * self.dataset_len):]

    def _get_allhrtf(self, norm_way=0):
        for idx in range(self.dataset_len):
            with open(os.path.join(self.hrir_path, f"{self.name}_{idx:03d}.pkl"), 'rb') as handle:
                location_sph, hrir = pkl.load(handle)  # location_sph: [B, 3], hrir: [2, T]
                location_sph = th.from_numpy(location_sph).float()
                hrir = th.from_numpy(hrir).float()

                if int(2 * self.max_f) == int(self.sampling_rate):
                    hrir_us = hrir
                else:
                    downsampler = ta.transforms.Resample(self.sampling_rate, int(2 * self.max_f))
                    hrir_us = downsampler(hrir)

                # Compute FFT and extract positive frequencies
                HRTF = th.fft.fft(hrir_us, n=256)
                HRTF_pm = th.conj(HRTF)
                HRTF = HRTF_pm[:, :, 1:self.filter_length + 1]  # shape: [2, B, 128]
                mag = torch.abs(HRTF) + 1e-8

                # Apply normalization
                if norm_way == 0:
                    scale_factor = torch.tensor(1, dtype=th.float32)
                elif norm_way == 1:
                    scale_factor = torch.max(mag)
                elif norm_way == 2:
                    mag_flatten = mag.flatten()
                    scale_factor = np.mean(sorted(mag_flatten.numpy())[-int(mag_flatten.numel() / 20):])
                    scale_factor = torch.tensor(scale_factor, dtype=th.float32)
                elif norm_way == 3:
                    equator_index = (location_sph[:, 1] >= -1) & (location_sph[:, 1] <= 0)
                    mag_equator = mag[equator_index]
                    equator_azi = location_sph[equator_index, 0]
                    sorted_idx = torch.argsort(equator_azi)
                    sorted_mag = mag_equator[sorted_idx]
                    total_energy = sum(torch.square(sorted_mag[x]).mean() *
                                       (360 if x == 0 else equator_azi[sorted_idx[x]] - equator_azi[sorted_idx[x-1]])
                                       for x in range(len(sorted_idx)))
                    scale_factor = torch.sqrt(total_energy / 360)

                HRTF_norm = HRTF / scale_factor

                if idx == 0:
                    locations_sph = th.zeros(self.dataset_len, location_sph.shape[0], 3)
                    HRTFs = th.zeros(self.dataset_len, HRTF.shape[0], HRTF.shape[1], HRTF.shape[2], dtype=th.complex64)
                locations_sph[idx] = location_sph
                HRTFs[idx] = HRTF_norm

        return HRTFs, locations_sph

    def __len__(self):
        return self.HRTFs_train.shape[0]

    def __getitem__(self, index):
        return self.locations_train[index], self.HRTFs_train[index]

    def trainitem(self):
        return self.locations_train, self.HRTFs_train

    def validitem(self):
        return self.locations_valid, self.HRTFs_valid

    def testitem(self):
        return self.locations_test, self.HRTFs_test

    def allitem(self):
        return self.locations, self.HRTFs


class MergedHRTFDataset(Dataset):
    def __init__(self, all_dataset_names, hrir_path="../preprocessed_data/HRIR",
                 scale="linear", norm_way=0, split="train"):
        """
        Args:
            all_dataset_names: list of dataset names to be merged
            hrir_path: root path to HRIR pkl files
            norm_way: normalization strategy to pass to HRTFDataset
            split: 'train' only used (validation/test not handled here)
        """
        self.all_dataset_names = all_dataset_names
        self.db_to_indices = {name: [] for name in all_dataset_names}
        self.hrir_path = hrir_path
        self.length_array = []
        self.all_data = []

        for dataset_name in self.all_dataset_names:
            dataset = HRTFDataset(dataset=dataset_name, hrir_path=hrir_path)
            for i in range(len(dataset)):
                locs, hrtfs = dataset[i]
                self.all_data.append((locs, hrtfs, dataset_name))
                self.db_to_indices[dataset_name].append(len(self.all_data) - 1)
            self.length_array.append(len(dataset))

        self.n_freq = self.all_data[0][1].shape[2]
        self.triplet_indices = self._construct_triplets(self.all_data, self.db_to_indices)

    def _construct_triplets(self, all_data, db_to_indices, num_triplets_per_anchor=1):
        """
        For each anchor, sample 1 positive from same db and 1 negative from a different db.
        """
        triplet_indices = []
        for anchor_idx, (_, _, anchor_db) in enumerate(all_data):
            pos_pool = [i for i in db_to_indices[anchor_db] if i != anchor_idx]
            if not pos_pool:
                continue
            neg_dbs = [db for db in db_to_indices if db != anchor_db]
            neg_db = random.choice(neg_dbs)
            neg_pool = db_to_indices[neg_db]
            for _ in range(num_triplets_per_anchor):
                pos_idx = random.choice(pos_pool)
                neg_idx = random.choice(neg_pool)
                triplet_indices.append((anchor_idx, pos_idx, neg_idx))
        return triplet_indices

    def __len__(self):
        return len(self.triplet_indices)

    def __getitem__(self, idx):
        a_idx, p_idx, n_idx = self.triplet_indices[idx]
        return self.all_data[a_idx], self.all_data[p_idx], self.all_data[n_idx]

    def collate_fn(self, triplet_batch):
        anchors, positives, negatives = zip(*triplet_batch)
        return self.pad_batch(anchors), self.pad_batch(positives), self.pad_batch(negatives)

    def pad_batch(self, batch_samples):
        B = len(batch_samples)
        max_locs = max(sample[0].shape[0] for sample in batch_samples)
        locs = -th.ones((B, max_locs, 3), dtype=th.float32)
        hrtfs = -th.ones((B, max_locs, 2, self.n_freq), dtype=th.complex64)
        masks = th.zeros((B, max_locs), dtype=th.float32)
        names = []

        for i, (loc, hrtf, name) in enumerate(batch_samples):
            n = loc.shape[0]
            locs[i, :n, :] = loc
            hrtfs[i, :n, :, :] = hrtf
            masks[i, :n] = 1
            names.append(name)
        return locs, hrtfs, masks, names




if __name__ == '__main__':
    print("=== Single Dataset Load Test ===")
    ds = HRTFDataset(dataset="ari")
    print("Train size:", len(ds))
    loc, hrtf = ds[0]
    print("Location shape:", loc.shape)  # (B, 3)
    print("HRTF shape:", hrtf.shape)  # (B, 2, 128)

    print("\n=== Merged Triplet Dataset Test ===")
    merged = MergedHRTFDataset(["ari", "bili","cipic"])
    print("Triplet count:", len(merged))
    anchor, pos, neg = merged[0]
    anchor_locs, anchor_hrtfs, anchor_name = anchor
    pos_locs, pos_hrtfs, pos_name = pos
    neg_locs, neg_hrtfs, neg_name = neg
    print("Anchor loc shape:", anchor_locs.shape, "Dataset:", anchor_name)
    print("Positive from same dataset:", pos_name, "Negative from:", neg_name)

    print("\n=== Collate Function Test ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(merged, batch_size=12, collate_fn=merged.collate_fn,shuffle=True)

    anchor_batch, positive_batch, negative_batch = next(iter(loader))

    anchor_locs, anchor_hrtfs, anchor_masks, anchor_names = anchor_batch
    print("Padded anchor locs shape:", anchor_locs.shape)
    print("Padded anchor HRTFs shape:", anchor_hrtfs.shape)
    print("Anchor masks shape:", anchor_masks.shape)
    print("Anchor names:", anchor_names)
