import argparse
import os
import pickle as pkl

import numpy as np
from scipy.spatial import KDTree


LAP_SPARSE_LEVELS = (3, 5, 19, 100)


def sph_deg_to_cart(azimuth_deg, elevation_deg):
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))

    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack([x, y, z], axis=-1)


def build_lap_target_grid(level):
    """Build the SONICOM/LAP sparse target directions in Cartesian coordinates."""
    if level == 3:
        az_all = [0, 90, 0]
        el_all = [0, 0, 90]
    elif level == 5:
        az_all = [0, 0, 0, 45, -45]
        el_all = [0, 90, -90, 0, 0]
    elif level == 19:
        az_list = [0, 60, 120, 180, 240, 300]
        el_list = [-45, 0, 45]
        az_all = [az for el in el_list for az in az_list]
        el_all = [el for el in el_list for _ in az_list]
        az_all.append(0)
        el_all.append(90)
    elif level == 100:
        elevs = [-45, -30, -15, 0, 15, 30, 45, 60, 75]
        azis = [0, 35, 70, 105, 140, 175, 210, 245, 280, 315, 350]
        az_all = [az for el in elevs for az in azis]
        el_all = [el for el in elevs for _ in azis]
        az_all.append(0)
        el_all.append(90)
    else:
        raise ValueError(f"Unsupported LAP sparse level: {level}")

    return sph_deg_to_cart(az_all, el_all)


def sofa_spherical_to_cart(locations):
    """Convert SOFA spherical positions [azimuth, elevation, radius] to unit xyz."""
    locations = np.asarray(locations, dtype=np.float64)
    return sph_deg_to_cart(locations[:, 0], locations[:, 1])


def nearest_lap_indices(locations, level, return_target=False):
    measured_xyz = sofa_spherical_to_cart(locations)
    measured_xyz = measured_xyz / np.linalg.norm(measured_xyz, axis=1, keepdims=True)
    target_xyz = build_lap_target_grid(level)

    _, idx = KDTree(measured_xyz).query(target_xyz)
    idx = sorted(set(idx.tolist()))

    if return_target:
        return idx, target_xyz
    return idx


def save_sparse_index_file(hrir_directory, output_file, levels=LAP_SPARSE_LEVELS):
    sparse_indices = {}
    for filename in sorted(os.listdir(hrir_directory)):
        if not filename.endswith(".pkl"):
            continue
        with open(os.path.join(hrir_directory, filename), "rb") as handle:
            locations, _ = pkl.load(handle)
        sparse_indices[filename] = {
            level: nearest_lap_indices(locations, level) for level in levels
        }

    with open(output_file, "wb") as handle:
        pkl.dump(sparse_indices, handle, protocol=pkl.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrir_directory", default="../preprocessed_data/HRIR")
    parser.add_argument("--output_file", default="../preprocessed_data/lap_sparse_indices.pkl")
    args = parser.parse_args()
    save_sparse_index_file(args.hrir_directory, args.output_file)


if __name__ == "__main__":
    main()
