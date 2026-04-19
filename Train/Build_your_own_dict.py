#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import pickle
import argparse
import anndata as ad
import numpy as np
import torch
import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(
        "Build feature_dict, data_dict_processed, and spatial_loc_dict "
        "from user-organized h5ad files."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Project root containing sections.txt, HE/, RNA/, Protein/."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <project-root>/Data_preprocessing"
    )
    parser.add_argument(
        "--he-feature-key",
        type=str,
        default="HE_harmony",
        help="obsm key used as HE feature. Default: HE_harmony"
    )
    parser.add_argument(
        "--rna-feature-key",
        type=str,
        default="RNA_harmony",
        help="obsm key used as RNA feature. Default: RNA_harmony"
    )
    parser.add_argument(
        "--protein-feature-key",
        type=str,
        default="Protein_harmony",
        help="obsm key used as Protein feature. Default: Protein_harmony"
    )
    parser.add_argument(
        "--feature-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Tensor dtype for feature_dict. Default: float32"
    )

    return parser.parse_args()


def read_sections(section_txt_path):
    if not os.path.exists(section_txt_path):
        raise FileNotFoundError(f"sections.txt not found: {section_txt_path}")

    sections = []
    with open(section_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            sections.append(line)

    if len(sections) == 0:
        raise ValueError("sections.txt is empty.")

    if len(set(sections)) != len(sections):
        raise ValueError("sections.txt contains duplicated section names.")

    return sections


def get_tensor_dtype(dtype_str):
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def load_modality_h5ad(modality_dir, section):
    fpath = os.path.join(modality_dir, f"adata_{section}.h5ad")
    if not os.path.exists(fpath):
        return None, fpath
    adata = sc.read_h5ad(fpath)
    return adata, fpath


def check_spatial(adata, section, modality):
    if "spatial" not in adata.obsm:
        raise KeyError(f"{modality}/{section}.h5ad missing obsm['spatial']")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError(
            f"{modality}/{section}.h5ad has invalid obsm['spatial'] shape: {spatial.shape}"
        )
    if spatial.shape[0] != adata.n_obs:
        raise ValueError(
            f"{modality}/{section}.h5ad row mismatch: "
            f"n_obs={adata.n_obs}, spatial rows={spatial.shape[0]}"
        )
    return spatial


def check_feature(adata, section, modality, feature_key):
    if feature_key not in adata.obsm:
        raise KeyError(f"{modality}/{section}.h5ad missing obsm['{feature_key}']")
    feat = np.asarray(adata.obsm[feature_key], dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError(
            f"{modality}/{section}.h5ad has invalid obsm['{feature_key}'] shape: {feat.shape}"
        )
    if feat.shape[0] != adata.n_obs:
        raise ValueError(
            f"{modality}/{section}.h5ad row mismatch: "
            f"n_obs={adata.n_obs}, feature rows={feat.shape[0]}"
        )
    return feat


def check_molecular_var_names(adata, section, modality):
    if adata.var_names is None or len(adata.var_names) != adata.n_vars:
        raise ValueError(f"{modality}/{section}.h5ad has invalid var_names")
    if len(set(map(str, adata.var_names))) != len(adata.var_names):
        adata.var_names_make_unique()


def assert_cross_modality_consistency(section, adata_by_mod):
    present = {k: v for k, v in adata_by_mod.items() if v is not None}
    if len(present) <= 1:
        return

    mods = list(present.keys())
    ref_mod = mods[0]
    ref_adata = present[ref_mod]
    ref_n_obs = ref_adata.n_obs
    ref_spatial = check_spatial(ref_adata, section, ref_mod)

    for mod in mods[1:]:
        adata = present[mod]
        spatial = check_spatial(adata, section, mod)

        if adata.n_obs != ref_n_obs:
            raise ValueError(
                f"Inconsistent n_obs in section {section}: "
                f"{ref_mod} has {ref_n_obs}, but {mod} has {adata.n_obs}"
            )

        if spatial.shape != ref_spatial.shape:
            raise ValueError(
                f"Inconsistent spatial shape in section {section}: "
                f"{ref_mod} has {ref_spatial.shape}, but {mod} has {spatial.shape}"
            )

        if not np.allclose(spatial, ref_spatial):
            raise ValueError(
                f"Inconsistent spatial coordinates in section {section}: "
                f"{ref_mod} and {mod} have different obsm['spatial']"
            )


def main():
    args = parse_args()

    project_root = args.project_root
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(project_root, "Data_preprocessing")

    os.makedirs(output_dir, exist_ok=True)

    sections_txt = os.path.join(project_root, "sections.txt")
    he_dir = os.path.join(project_root, "HE")
    rna_dir = os.path.join(project_root, "RNA")
    protein_dir = os.path.join(project_root, "Protein")

    sections = read_sections(sections_txt)
    tensor_dtype = get_tensor_dtype(args.feature_dtype)

    print("\n" + "=" * 80)
    print("Step 1. Read section order")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Total sections: {len(sections)}")
    print("Section order:", sections)

    feature_dict = {}
    spatial_loc_dict = {}
    data_dict_processed = {
        "HE": [],
        "RNA": [],
        "Protein": [],
    }
    adata_he_list = []

    print("\n" + "=" * 80)
    print("Step 2. Build dictionaries from user h5ad files")
    print("=" * 80)

    for section in sections:
        print(f"\nProcessing {section}")

        adata_he, he_path = load_modality_h5ad(he_dir, section)
        adata_rna, rna_path = load_modality_h5ad(rna_dir, section)
        adata_protein, protein_path = load_modality_h5ad(protein_dir, section)

        adata_by_mod = {
            "HE": adata_he,
            "RNA": adata_rna,
            "Protein": adata_protein,
        }

        if all(v is None for v in adata_by_mod.values()):
            raise ValueError(f"Section {section} is missing in all modalities")

        assert_cross_modality_consistency(section, adata_by_mod)

        feature_dict[section] = {}

        # if adata_he is not None:
        #     spatial_he = check_spatial(adata_he, section, "HE")
        #     feat_he = check_feature(adata_he, section, "HE", args.he_feature_key)
        #     feature_dict[section]["HE"] = torch.tensor(feat_he, dtype=tensor_dtype)
        #     data_dict_processed["HE"].append(adata_he)
        #     print(
        #         f"  HE: n_obs={adata_he.n_obs}, n_vars={adata_he.n_vars}, "
        #         f"feature={feat_he.shape}"
        #     )
        # else:
        #     data_dict_processed["HE"].append(None)
        #     print("  HE: missing")

        if adata_he is not None:
            spatial_he = check_spatial(adata_he, section, "HE")
            feat_he = check_feature(adata_he, section, "HE", args.he_feature_key)
        
            feature_dict[section]["HE"] = torch.tensor(feat_he, dtype=tensor_dtype)
            data_dict_processed["HE"].append(adata_he)
        
            # Build HE-only AnnData
            adata_he_only = ad.AnnData(X=np.asarray(feat_he, dtype=np.float32))
            adata_he_only.obsm["spatial"] = spatial_he
        
            try:
                adata_he_only.obs_names = adata_he.obs_names.copy()
            except Exception:
                print("  [Warning] Failed to copy HE obs_names, continue without it.")
        
            adata_he_list.append(adata_he_only)
        
            print(
                f"  HE: n_obs={adata_he.n_obs}, n_vars={adata_he.n_vars}, "
                f"feature={feat_he.shape}"
            )
        else:
            data_dict_processed["HE"].append(None)
            adata_he_list.append(None)
            print("  HE: missing")

        if adata_rna is not None:
            spatial_rna = check_spatial(adata_rna, section, "RNA")
            check_molecular_var_names(adata_rna, section, "RNA")
            feat_rna = check_feature(adata_rna, section, "RNA", args.rna_feature_key)
            feature_dict[section]["RNA"] = torch.tensor(feat_rna, dtype=tensor_dtype)
            data_dict_processed["RNA"].append(adata_rna)
            print(
                f"  RNA: n_obs={adata_rna.n_obs}, n_vars={adata_rna.n_vars}, "
                f"feature={feat_rna.shape}"
            )
        else:
            data_dict_processed["RNA"].append(None)
            print("  RNA: missing")

        if adata_protein is not None:
            spatial_protein = check_spatial(adata_protein, section, "Protein")
            check_molecular_var_names(adata_protein, section, "Protein")
            feat_protein = check_feature(
                adata_protein, section, "Protein", args.protein_feature_key
            )
            feature_dict[section]["Protein"] = torch.tensor(feat_protein, dtype=tensor_dtype)
            data_dict_processed["Protein"].append(adata_protein)
            print(
                f"  Protein: n_obs={adata_protein.n_obs}, n_vars={adata_protein.n_vars}, "
                f"feature={feat_protein.shape}"
            )
        else:
            data_dict_processed["Protein"].append(None)
            print("  Protein: missing")

        if adata_he is not None:
            spatial_loc_dict[section] = np.asarray(adata_he.obsm["spatial"], dtype=np.float32)
        elif adata_rna is not None:
            spatial_loc_dict[section] = np.asarray(adata_rna.obsm["spatial"], dtype=np.float32)
        else:
            spatial_loc_dict[section] = np.asarray(adata_protein.obsm["spatial"], dtype=np.float32)

        print(f"  spatial: {spatial_loc_dict[section].shape}")

        gc.collect()

    print("\n" + "=" * 80)
    print("Step 3. Save outputs")
    print("=" * 80)

    feature_concat_path = os.path.join(output_dir, "feature_dict_concat.pkl")
    data_processed_concat_path = os.path.join(output_dir, "data_dict_processed_concat.pkl")
    spatial_loc_dict_path = os.path.join(output_dir, "spatial_loc_dict.pkl")
    he_output_dir = os.path.join(output_dir, "HE")
    os.makedirs(he_output_dir, exist_ok=True)
    
    he_only_pkl = os.path.join(he_output_dir, "data_dict_HE_only.pkl")

    with open(feature_concat_path, "wb") as f:
        pickle.dump(feature_dict, f)
    print(f"Saved: {feature_concat_path}")

    with open(data_processed_concat_path, "wb") as f:
        pickle.dump(data_dict_processed, f)
    print(f"Saved: {data_processed_concat_path}")

    with open(spatial_loc_dict_path, "wb") as f:
        pickle.dump(spatial_loc_dict, f)
    print(f"Saved: {spatial_loc_dict_path}")

    with open(he_only_pkl, "wb") as f:
        pickle.dump({"HE": adata_he_list}, f)
    print(f"Saved: {he_only_pkl}")

    print("\n" + "=" * 80)
    print("Step 4. Summary")
    print("=" * 80)

    n_he = sum(x is not None for x in data_dict_processed["HE"])
    n_rna = sum(x is not None for x in data_dict_processed["RNA"])
    n_protein = sum(x is not None for x in data_dict_processed["Protein"])

    print(f"Sections total:    {len(sections)}")
    print(f"HE available:      {n_he}")
    print(f"RNA available:     {n_rna}")
    print(f"Protein available: {n_protein}")

    print("\nAll dictionaries were built successfully.")


if __name__ == "__main__":
    main()