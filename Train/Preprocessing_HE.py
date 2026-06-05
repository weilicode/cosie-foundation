#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import pickle
import joblib
import argparse

import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import setup_seed



# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("HE preprocessing pipeline")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--run-metacell",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
        help="Whether to run metacell (default: True)"
    )
    return parser.parse_args()


def find_he_paths(data_root):
    """
    Automatically find HE h5ad files under <data_root>/HE/.

    Expected filename pattern
    -------------------------
    adata_<section_name>.h5ad

    Returns
    -------
    list of str
        Sorted HE h5ad paths.
    """
    he_dir = os.path.join(data_root, "HE")
    if not os.path.exists(he_dir):
        raise FileNotFoundError(f"HE directory not found: {he_dir}")

    he_paths = [
        os.path.join(he_dir, x)
        for x in os.listdir(he_dir)
        if x.endswith(".h5ad") and x.startswith("adata_")
    ]

    if len(he_paths) == 0:
        raise ValueError(f"No HE h5ad files found under: {he_dir}")

    he_paths = sorted(he_paths)
    return he_paths


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    # =========================================================
    # 0. Basic settings
    # =========================================================
    setup_seed(0)

    n_components = 50
    batch_size = 50000
    sample_rate = 0.05
    run_metacell = args.run_metacell

    # =========================================================
    # 1. Paths
    # =========================================================
    data_root = args.data_root

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    parent_dir = os.path.dirname(data_root.rstrip("/"))
    he_preprocess_dir = os.path.join(parent_dir, "Data_preprocessing", "HE")
    os.makedirs(he_preprocess_dir, exist_ok=True)

    save_mc_path = os.path.join(he_preprocess_dir, "meta_cell_2_2")
    os.makedirs(save_mc_path, exist_ok=True)

    # =========================================================
    # 2. HE paths
    # =========================================================
    he_paths = find_he_paths(data_root)

    print("Total HE sections:", len(he_paths))

    section_names = [
        os.path.basename(p).replace("adata_", "").replace(".h5ad", "")
        for p in he_paths
    ]

    print("HE section names:")
    for x in section_names:
        print("  ", x)

    # save section order for downstream reference
    section_order_path = os.path.join(he_preprocess_dir, "all_section_names.pkl")
    with open(section_order_path, "wb") as f:
        pickle.dump(section_names, f)
    print(f"Saved section order to: {section_order_path}")

    # =========================================================
    # 3. Step 1: PCA fitting
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Fit IncrementalPCA on HE UNI_feature")
    print("=" * 80)

    pca = IncrementalPCA(n_components=n_components)

    for i, path in enumerate(he_paths, start=1):
        print(f"\n[{i}/{len(he_paths)}] Processing for PCA fitting:")
        print(path)

        if not os.path.exists(path):
            print(f"  [Skip] File not found: {path}")
            continue

        adata = sc.read_h5ad(path, backed="r")
        X = adata.file["obsm"]["UNI_feature"]

        n_cells = X.shape[0]
        n_features = X.shape[1]

        print(f"  UNI_feature shape = ({n_cells}, {n_features})")

        n_samples = int(n_cells * sample_rate)
        if n_samples < batch_size:
            n_samples = min(batch_size, n_cells)

        idx = np.random.choice(n_cells, n_samples, replace=False)
        idx.sort()

        print(f"  Sampled {n_samples:,} / {n_cells:,} cells")

        for start in tqdm(range(0, n_samples, batch_size), desc=f"  Fitting {section_names[i-1]}"):
            end = min(start + batch_size, n_samples)
            X_batch = np.array(X[idx[start:end]], dtype=np.float32)

            if X_batch.shape[0] >= n_components:
                pca.partial_fit(X_batch)
            else:
                print(f"  [Warning] Skip small batch: {X_batch.shape[0]} < n_components={n_components}")

            del X_batch
            gc.collect()

        adata.file.close()
        del adata, X
        gc.collect()

    print("\n[Done] PCA fitting finished.")

    # =========================================================
    # 4. Save PCA
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Save PCA model")
    print("=" * 80)

    pca_npz_path = os.path.join(he_preprocess_dir, "joint_HE_PCA_model.npz")
    np.savez_compressed(
        pca_npz_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        n_components=n_components,
    )
    print(f"Saved PCA params to: {pca_npz_path}")

    pca_joblib_path = os.path.join(he_preprocess_dir, "joint_HE_PCA_model.joblib")
    joblib.dump(pca, pca_joblib_path)
    print(f"Saved PCA joblib to: {pca_joblib_path}")

    pca_pkl_path = os.path.join(he_preprocess_dir, "joint_HE_PCA_model.pkl")
    with open(pca_pkl_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA pickle to: {pca_pkl_path}")

    # =========================================================
    # 5. Transform
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Transform each HE section with trained PCA")
    print("=" * 80)

    for i, (path, name) in enumerate(zip(he_paths, section_names), start=1):
        print(f"\n[{i}/{len(he_paths)}] Transforming:")
        print(path)

        if not os.path.exists(path):
            print(f"  [Skip] File not found: {path}")
            continue

        adata = sc.read_h5ad(path, backed="r")
        X = adata.file["obsm"]["UNI_feature"]
        n_cells = X.shape[0]
        n_features = X.shape[1]

        print(f"  UNI_feature shape = ({n_cells}, {n_features})")

        X_full = np.array(X, dtype=np.float32)
        print(f"  Loaded full feature matrix into memory: {X_full.shape}, {X_full.dtype}")

        X_pca = pca.transform(X_full).astype(np.float16)
        print(f"  PCA transformed shape = {X_pca.shape}, dtype = {X_pca.dtype}")

        del X_full
        gc.collect()

        out_name = f"adata_{name}_HE_PCA_50d.npy"
        out_path = os.path.join(he_preprocess_dir, out_name)
        np.save(out_path, X_pca)

        print(f"  Saved PCA embedding to: {out_path}")

        adata.file.close()
        del adata, X, X_pca
        gc.collect()

    print("\n[Done] All HE PCA files saved.")

    # =========================================================
    # 6. Build HE-only AnnData
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Build HE-only AnnData list")
    print("=" * 80)

    adata_he_list = []

    for i, (path, name) in enumerate(zip(he_paths, section_names), start=1):
        print(f"\n[{i}/{len(he_paths)}] Building HE AnnData:")
        print(path)

        if not os.path.exists(path):
            print(f"  [Skip] Original h5ad not found: {path}")
            adata_he_list.append(None)
            continue

        npy_path = os.path.join(he_preprocess_dir, f"adata_{name}_HE_PCA_50d.npy")
        if not os.path.exists(npy_path):
            print(f"  [Skip] PCA npy not found: {npy_path}")
            adata_he_list.append(None)
            continue

        adata_backed = sc.read_h5ad(path, backed="r")
        spatial = np.asarray(adata_backed.obsm["spatial"], dtype=np.float32)
        n_obs_spatial = spatial.shape[0]

        HE_pca = np.load(npy_path).astype(np.float32, copy=False)
        n_obs_pca = HE_pca.shape[0]

        print(f"  spatial shape = {spatial.shape}")
        print(f"  HE PCA shape  = {HE_pca.shape}")

        if n_obs_spatial != n_obs_pca:
            raise ValueError(
                f"Shape mismatch for {name}: spatial n_obs={n_obs_spatial}, HE_pca n_obs={n_obs_pca}"
            )

        adata_he = ad.AnnData(X=HE_pca)
        adata_he.obsm["spatial"] = spatial

        try:
            adata_he.obs_names = adata_backed.obs_names.copy()
        except Exception:
            print("  [Warning] Failed to copy obs_names, continue without it.")

        adata_he_list.append(adata_he)
        print(f"  Created HE AnnData with shape: {adata_he.shape}")

        adata_backed.file.close()
        del adata_backed, spatial, HE_pca, adata_he
        gc.collect()

    print("\n[Done] HE-only AnnData list built.")

    # =========================================================
    # 7. Save HE-only dict
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Save HE-only data_dict")
    print("=" * 80)

    he_only_pkl = os.path.join(he_preprocess_dir, "data_dict_HE_only.pkl")
    with open(he_only_pkl, "wb") as f:
        pickle.dump({"HE": adata_he_list}, f)

    print(f"Saved HE-only data_dict to: {he_only_pkl}")

    # =========================================================
    # 8. Metacell
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Optional metacell construction")
    print("=" * 80)

    if run_metacell:
        print("Metacell construction is ENABLED.")

        for section_name, adata_he in zip(section_names, adata_he_list):
            print(f"\n▶ Processing HE section: {section_name}")

            if adata_he is None:
                print("  [Skip] This section is None.")
                continue

            adata_mc = metacell_construction_optimized(adata_he)

            meta_map = adata_mc.uns["meta_to_original"]
            map_path = os.path.join(save_mc_path, f"meta_to_original_{section_name}.pkl")
            with open(map_path, "wb") as f:
                pickle.dump(meta_map, f)
            print(f"  Saved metacell mapping to: {map_path}")

            del adata_mc.uns["meta_to_original"]

            mc_h5ad_path = os.path.join(save_mc_path, f"{section_name}_he_mc.h5ad")
            adata_mc.write(mc_h5ad_path)
            print(f"  Saved metacell h5ad to: {mc_h5ad_path}")
            print(f"  Metacell shape = {adata_mc.shape}")

            del adata_mc, meta_map
            gc.collect()

        print("\n[Done] HE metacell construction completed.")

    else:
        print("Metacell construction is DISABLED. Skipping this step.")

    # =========================================================
    # 9. Prepare data_dict
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Prepare data_dict for load_data")
    print("=" * 80)

    if run_metacell:
        adata_he_for_preprocessing = []
        missing = []

        for section_name in section_names:
            path = os.path.join(save_mc_path, f"{section_name}_he_mc.h5ad")
            if os.path.exists(path):
                print(f"Loading metacell HE: {path}")
                adata_he_for_preprocessing.append(sc.read_h5ad(path))
            else:
                print(f"[Warning] Metacell file missing: {path}")
                adata_he_for_preprocessing.append(None)
                missing.append(section_name)

        if len(missing) > 0:
            raise FileNotFoundError(f"Missing metacell files: {missing}")

    else:
        data_dict_HE_only = pickle.load(open(he_only_pkl, "rb"))
        adata_he_for_preprocessing = data_dict_HE_only["HE"]
        print(f"Loaded non-metacell HE sections: {len(adata_he_for_preprocessing)}")

    data_dict = {"HE": adata_he_for_preprocessing}

    # =========================================================
    # 10. load_data
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 8. Run load_data preprocessing")
    print("=" * 80)

    feature_dict, spatial_loc_dict, data_dict_processed = load_data(
        data_dict,
        n_comps=50,
        metacell=False
    )

    print("Preprocessing finished.")
    print("Keys in feature_dict:", feature_dict.keys())
    print("Keys in spatial_loc_dict:", spatial_loc_dict.keys())
    print("Keys in data_dict_processed:", data_dict_processed.keys())

    # =========================================================
    # 11. Save outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 9. Save preprocessing outputs")
    print("=" * 80)

    feature_dict_path = os.path.join(he_preprocess_dir, "feature_dict_HE.pkl")
    spatial_loc_dict_path = os.path.join(he_preprocess_dir, "spatial_loc_dict_HE.pkl")
    data_dict_processed_path = os.path.join(he_preprocess_dir, "data_dict_processed_HE.pkl")

    with open(feature_dict_path, "wb") as f:
        pickle.dump(feature_dict, f)
    print(f"Saved feature_dict to: {feature_dict_path}")

    with open(spatial_loc_dict_path, "wb") as f:
        pickle.dump(spatial_loc_dict, f)
    print(f"Saved spatial_loc_dict to: {spatial_loc_dict_path}")

    with open(data_dict_processed_path, "wb") as f:
        pickle.dump(data_dict_processed, f)
    print(f"Saved data_dict_processed to: {data_dict_processed_path}")

    print("\n" + "=" * 80)
    print("All HE preprocessing steps completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()