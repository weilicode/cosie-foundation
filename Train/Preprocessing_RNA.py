#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import pickle
import argparse

import numpy as np
import scanpy as sc
import anndata as ad

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import setup_seed
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.downstream_analysis import *



# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("RNA preprocessing pipeline")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--run-metacell",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
        help="Whether to run metacell (default: True)"
    )
    return parser.parse_args()


def find_modality_paths(data_root, modality):
    """
    Automatically find h5ad files under <data_root>/<modality>/.

    Expected filename pattern
    -------------------------
    adata_<section_name>.h5ad

    Parameters
    ----------
    data_root : str
        Root directory containing modality folders.
    modality : str
        Modality folder name, e.g. 'HE', 'RNA', 'Protein'.

    Returns
    -------
    list of str
        Sorted h5ad paths.
    """
    modality_dir = os.path.join(data_root, modality)
    if not os.path.exists(modality_dir):
        raise FileNotFoundError(f"{modality} directory not found: {modality_dir}")

    paths = [
        os.path.join(modality_dir, x)
        for x in os.listdir(modality_dir)
        if x.endswith(".h5ad") and x.startswith("adata_")
    ]

    paths = sorted(paths)
    return paths


def get_section_names_from_paths(paths):
    return [
        os.path.basename(p).replace("adata_", "").replace(".h5ad", "")
        for p in paths
    ]


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    # =========================================================
    # 0. Basic settings
    # =========================================================
    setup_seed(0)
    run_metacell = args.run_metacell

    # =========================================================
    # 1. Define paths
    # =========================================================
    data_root = args.data_root

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    parent_dir = os.path.dirname(data_root.rstrip("/"))
    rna_preprocess_dir = os.path.join(parent_dir, "Data_preprocessing", "RNA")
    os.makedirs(rna_preprocess_dir, exist_ok=True)

    save_mc_path = os.path.join(rna_preprocess_dir, "meta_cell_2_2")
    os.makedirs(save_mc_path, exist_ok=True)

    # =========================================================
    # 2. Read master order from HE folder
    # =========================================================
    he_paths = find_modality_paths(data_root, "HE")
    if len(he_paths) == 0:
        raise ValueError(f"No HE files found under: {os.path.join(data_root, 'HE')}")

    all_section_names = get_section_names_from_paths(he_paths)

    print("Total master sections from HE:", len(all_section_names))
    print("Master section names:")
    for x in all_section_names:
        print("  ", x)

    master_section_path = os.path.join(rna_preprocess_dir, "all_section_names.pkl")
    with open(master_section_path, "wb") as f:
        pickle.dump(all_section_names, f)
    print(f"Saved master section order to: {master_section_path}")

    # =========================================================
    # 3. Read RNA section paths
    # =========================================================
    rna_paths = find_modality_paths(data_root, "RNA")

    if len(rna_paths) == 0:
        raise ValueError(f"No RNA files found under: {os.path.join(data_root, 'RNA')}")

    print("Total RNA files:", len(rna_paths))

    rna_section_names = get_section_names_from_paths(rna_paths)

    print("RNA section names:")
    for x in rna_section_names:
        print("  ", x)

    # =========================================================
    # 4. Build mapping from RNA section name to master section name
    # =========================================================
    # Since filenames are unified as adata_<section_name>.h5ad,
    # mapping is simply identity matching by section name.
    master_section_set = set(all_section_names)

    rna_to_master_name = {}
    unmatched_rna_sections = []

    for rna_name in rna_section_names:
        if rna_name in master_section_set:
            rna_to_master_name[rna_name] = rna_name
        else:
            unmatched_rna_sections.append(rna_name)

    print("\nRNA to master section mapping:")
    for k, v in rna_to_master_name.items():
        print(f"  {k}  -->  {v}")

    if len(unmatched_rna_sections) > 0:
        print("\n[Warning] These RNA sections could not be matched to HE master order:")
        for x in unmatched_rna_sections:
            print("  ", x)

    # =========================================================
    # 5. Optional metacell construction
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Optional metacell construction for RNA")
    print("=" * 80)

    if run_metacell:
        print("Metacell construction is ENABLED.")

        for i, (path, section_name) in enumerate(zip(rna_paths, rna_section_names), start=1):
            print(f"\n[{i}/{len(rna_paths)}] Processing RNA section:")
            print(path)

            if not os.path.exists(path):
                print(f"  [Skip] File not found: {path}")
                continue

            adata_rna = sc.read_h5ad(path)
            print(f"  Loaded RNA AnnData with shape: {adata_rna.shape}")

            if "UNI_feature" in adata_rna.obsm:
                del adata_rna.obsm["UNI_feature"]

            adata_rna.var_names_make_unique()

            adata_rna_mc = metacell_construction_optimized(adata_rna)

            meta_map = adata_rna_mc.uns["meta_to_original"]
            map_path = os.path.join(save_mc_path, f"meta_to_original_{section_name}.pkl")
            with open(map_path, "wb") as f:
                pickle.dump(meta_map, f)
            print(f"  Saved metacell mapping to: {map_path}")

            del adata_rna_mc.uns["meta_to_original"]

            adata_rna_mc.var_names_make_unique()

            mc_h5ad_path = os.path.join(save_mc_path, f"{section_name}_rna_mc.h5ad")
            adata_rna_mc.write(mc_h5ad_path)
            print(f"  Saved metacell h5ad to: {mc_h5ad_path}")
            print(f"  Metacell shape = {adata_rna_mc.shape}")

            del adata_rna, adata_rna_mc, meta_map
            gc.collect()

        print("\n[Done] RNA metacell construction completed.")

    else:
        print("Metacell construction is DISABLED. Skipping this step.")

    # =========================================================
    # 6. Prepare RNA data_dict in master order
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Prepare RNA data_dict for load_data")
    print("=" * 80)

    rna_dict_by_master_section = {}

    if run_metacell:
        print("Using metacell RNA for preprocessing.")

        for rna_name in rna_section_names:
            if rna_name not in rna_to_master_name:
                continue

            master_name = rna_to_master_name[rna_name]
            mc_h5ad_path = os.path.join(save_mc_path, f"{rna_name}_rna_mc.h5ad")

            if os.path.exists(mc_h5ad_path):
                print(f"Loading metacell RNA: {mc_h5ad_path}")
                adata_tmp = sc.read_h5ad(mc_h5ad_path)
                adata_tmp.var_names_make_unique()
                rna_dict_by_master_section[master_name] = adata_tmp
            else:
                print(f"[Warning] Missing metacell RNA file: {mc_h5ad_path}")

    else:
        print("Using original RNA for preprocessing.")

        for path, rna_name in zip(rna_paths, rna_section_names):
            if rna_name not in rna_to_master_name:
                continue

            print(f"Loading original RNA: {path}")

            if not os.path.exists(path):
                print(f"[Warning] Missing original RNA file: {path}")
                continue

            master_name = rna_to_master_name[rna_name]

            adata_tmp = sc.read_h5ad(path)

            if "UNI_feature" in adata_tmp.obsm:
                del adata_tmp.obsm["UNI_feature"]

            adata_tmp.var_names_make_unique()
            rna_dict_by_master_section[master_name] = adata_tmp

    adata_rna_for_preprocessing = []
    missing_sections = []

    for section_name in all_section_names:
        if section_name in rna_dict_by_master_section:
            adata_rna_for_preprocessing.append(rna_dict_by_master_section[section_name])
        else:
            adata_rna_for_preprocessing.append(None)
            missing_sections.append(section_name)

    print(
        f"\nValid RNA sections: "
        f"{sum(x is not None for x in adata_rna_for_preprocessing)} / {len(all_section_names)}"
    )
    print(f"Missing RNA sections automatically filled with None: {len(missing_sections)}")

    if len(missing_sections) > 0:
        print("Missing section names:")
        for x in missing_sections:
            print("  ", x)

    data_dict = {
        "RNA": adata_rna_for_preprocessing,
    }

    # =========================================================
    # 7. Run load_data preprocessing
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Run load_data preprocessing for RNA")
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
    # 8. Save preprocessing outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Save RNA preprocessing outputs")
    print("=" * 80)

    feature_dict_path = os.path.join(rna_preprocess_dir, "feature_dict_RNA.pkl")
    spatial_loc_dict_path = os.path.join(rna_preprocess_dir, "spatial_loc_dict_RNA.pkl")
    data_dict_processed_path = os.path.join(rna_preprocess_dir, "data_dict_processed_RNA.pkl")

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
    print("All RNA preprocessing steps completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()