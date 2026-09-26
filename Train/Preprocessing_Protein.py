#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, gc, pickle, argparse
import scanpy as sc

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser("Protein preprocessing pipeline")
    parser.add_argument("--data-root", type=str, required=True)
    return parser.parse_args()


def find_modality_paths(data_root, modality):
    modality_dir = os.path.join(data_root, modality)
    if not os.path.exists(modality_dir):
        raise FileNotFoundError(f"{modality} directory not found: {modality_dir}")

    return sorted([
        os.path.join(modality_dir, x)
        for x in os.listdir(modality_dir)
        if x.endswith(".h5ad") and x.startswith("adata_")
    ])


def get_section_names_from_paths(paths):
    return [
        os.path.basename(p).replace("adata_", "").replace(".h5ad", "")
        for p in paths
    ]


def main():
    args = parse_args()
    setup_seed(0)

    # =========================================================
    # 1. Paths
    # =========================================================
    data_root = args.data_root
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    parent_dir = os.path.dirname(data_root.rstrip("/"))
    protein_preprocess_dir = os.path.join(parent_dir, "Data_preprocessing", "Protein")
    os.makedirs(protein_preprocess_dir, exist_ok=True)

    # =========================================================
    # 2. Read master section order from HE
    # =========================================================
    he_paths = find_modality_paths(data_root, "HE")
    if len(he_paths) == 0:
        raise ValueError(f"No HE files found under: {os.path.join(data_root, 'HE')}")

    all_section_names = get_section_names_from_paths(he_paths)

    print("Total master sections from HE:", len(all_section_names))
    print("Master section names:", all_section_names)

    master_section_path = os.path.join(protein_preprocess_dir, "all_section_names.pkl")
    with open(master_section_path, "wb") as f:
        pickle.dump(all_section_names, f)

    # =========================================================
    # 3. Read Protein sections
    # =========================================================
    protein_paths = find_modality_paths(data_root, "Protein")
    if len(protein_paths) == 0:
        raise ValueError(f"No Protein files found under: {os.path.join(data_root, 'Protein')}")

    protein_section_names = get_section_names_from_paths(protein_paths)

    print("Total Protein files:", len(protein_paths))
    print("Protein section names:", protein_section_names)

    # =========================================================
    # 4. Prepare Protein data_dict
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Prepare Protein data_dict")
    print("=" * 80)

    master_section_set = set(all_section_names)
    protein_dict_by_section = {}

    for path, section_name in zip(protein_paths, protein_section_names):
        if section_name not in master_section_set:
            print(f"[Warning] Protein section not found in HE master order: {section_name}")
            continue

        print(f"Loading Protein: {path}")
        adata_tmp = sc.read_h5ad(path)

        if "UNI_feature" in adata_tmp.obsm:
            del adata_tmp.obsm["UNI_feature"]

        adata_tmp.var_names_make_unique()
        protein_dict_by_section[section_name] = adata_tmp

    adata_protein_for_preprocessing = []
    missing_sections = []

    for section_name in all_section_names:
        if section_name in protein_dict_by_section:
            adata_protein_for_preprocessing.append(protein_dict_by_section[section_name])
        else:
            adata_protein_for_preprocessing.append(None)
            missing_sections.append(section_name)

    print(
        f"Valid Protein sections: "
        f"{sum(x is not None for x in adata_protein_for_preprocessing)} / {len(all_section_names)}"
    )

    if missing_sections:
        print("Missing Protein sections:", missing_sections)

    data_dict = {"Protein": adata_protein_for_preprocessing}

    # =========================================================
    # 5. Run load_data preprocessing
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Run Protein preprocessing")
    print("=" * 80)

    feature_dict, spatial_loc_dict, data_dict_processed = load_data(
        data_dict,
        n_comps=50,
    )

    print("Preprocessing finished.")
    print("Keys in feature_dict:", feature_dict.keys())
    print("Keys in spatial_loc_dict:", spatial_loc_dict.keys())
    print("Keys in data_dict_processed:", data_dict_processed.keys())

    # =========================================================
    # 6. Save outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Save Protein preprocessing outputs")
    print("=" * 80)

    feature_dict_path = os.path.join(protein_preprocess_dir, "feature_dict_Protein.pkl")
    spatial_loc_dict_path = os.path.join(protein_preprocess_dir, "spatial_loc_dict_Protein.pkl")
    data_dict_processed_path = os.path.join(protein_preprocess_dir, "data_dict_processed_Protein.pkl")

    with open(feature_dict_path, "wb") as f:
        pickle.dump(feature_dict, f)

    with open(spatial_loc_dict_path, "wb") as f:
        pickle.dump(spatial_loc_dict, f)

    with open(data_dict_processed_path, "wb") as f:
        pickle.dump(data_dict_processed, f)

    print(f"Saved feature_dict to: {feature_dict_path}")
    print(f"Saved spatial_loc_dict to: {spatial_loc_dict_path}")
    print(f"Saved data_dict_processed to: {data_dict_processed_path}")

    del data_dict, data_dict_processed, feature_dict, spatial_loc_dict
    gc.collect()

    print("\n" + "=" * 80)
    print("All Protein preprocessing steps completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()