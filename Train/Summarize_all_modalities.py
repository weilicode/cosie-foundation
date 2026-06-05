#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")



# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("Merge HE / RNA / Protein preprocessing results")

    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing HE/, RNA/, Protein/ folders")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as data-root)")

    parser.add_argument("--verbose", type=int, default=1,
                        help="Print details (1: yes, 0: no)")

    return parser.parse_args()


# =========================================================
# Helper
# =========================================================
def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def vprint(msg, verbose):
    if verbose:
        print(msg)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    data_root = args.data_root
    output_dir = args.output_dir if args.output_dir else data_root
    verbose = args.verbose

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # 1. Load HE
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Load HE pkl")
    print("=" * 80)

    feature_dict_HE = load_pickle(os.path.join(data_root, "HE/feature_dict_HE.pkl"))
    data_dict_processed_HE = load_pickle(os.path.join(data_root, "HE/data_dict_processed_HE.pkl"))

    vprint(f"Loaded HE feature keys: {list(feature_dict_HE.keys())[:5]}...", verbose)

    # =========================================================
    # 2. Load RNA
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Load RNA pkl")
    print("=" * 80)

    feature_dict_RNA = load_pickle(os.path.join(data_root, "RNA/feature_dict_RNA.pkl"))
    data_dict_processed_RNA = load_pickle(os.path.join(data_root, "RNA/data_dict_processed_RNA.pkl"))

    vprint(f"Loaded RNA feature keys: {list(feature_dict_RNA.keys())[:5]}...", verbose)

    # =========================================================
    # 3. Load Protein
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Load Protein pkl")
    print("=" * 80)

    feature_dict_ADT = load_pickle(os.path.join(data_root, "Protein/feature_dict_Protein.pkl"))
    data_dict_processed_ADT = load_pickle(os.path.join(data_root, "Protein/data_dict_processed_Protein.pkl"))

    vprint(f"Loaded Protein feature keys: {list(feature_dict_ADT.keys())[:5]}...", verbose)

    # =========================================================
    # 4. Merge feature_dict
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Merge feature_dict")
    print("=" * 80)

    merged_feature_dict = defaultdict(dict)

    for fd in [feature_dict_HE, feature_dict_RNA, feature_dict_ADT]:
        for sec, mod_dict in fd.items():
            merged_feature_dict[sec].update(mod_dict)

    feature_dict = dict(merged_feature_dict)

    print("Merged feature_dict sections:", len(feature_dict))

    # =========================================================
    # 5. Merge data_dict_processed
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Merge data_dict_processed")
    print("=" * 80)

    data_dict_processed = {}

    if "HE" in data_dict_processed_HE:
        data_dict_processed["HE"] = data_dict_processed_HE["HE"]
    else:
        print("[Warning] 'HE' missing")

    if "RNA" in data_dict_processed_RNA:
        data_dict_processed["RNA"] = data_dict_processed_RNA["RNA"]
    else:
        print("[Warning] 'RNA' missing")

    if "Protein" in data_dict_processed_ADT:
        data_dict_processed["Protein"] = data_dict_processed_ADT["Protein"]
    else:
        print("[Warning] 'Protein' missing")

    print("Merged modalities:", data_dict_processed.keys())

    # =========================================================
    # 6. Sanity check
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Sanity checks")
    print("=" * 80)

    all_sections = sorted(feature_dict.keys())
    print("Total sections:", len(all_sections))

    if verbose:
        for sec in all_sections:
            mods = list(feature_dict[sec].keys())
            print(f"{sec}: {mods}")

    print("\nSection modality summary:")
    for sec in all_sections:
        print(
            f"{sec}: "
            f"HE={'HE' in feature_dict[sec]}, "
            f"RNA={'RNA' in feature_dict[sec]}, "
            f"Protein={'Protein' in feature_dict[sec]}"
        )

    # =========================================================
    # 7. Save outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Save merged outputs")
    print("=" * 80)

    feature_out = os.path.join(output_dir, "feature_dict_concat.pkl")
    data_processed_out = os.path.join(output_dir, "data_dict_processed_concat.pkl")
    spatial_loc_out = os.path.join(output_dir, "spatial_loc_dict.pkl")

    with open(feature_out, "wb") as f:
        pickle.dump(feature_dict, f)

    with open(data_processed_out, "wb") as f:
        pickle.dump(data_dict_processed, f)

    spatial_dict = load_pickle(os.path.join(data_root, "HE/spatial_loc_dict_HE.pkl"))
    with open(spatial_loc_out, "wb") as f:
        pickle.dump(spatial_dict, f)

    print(f"Saved feature_dict to: {feature_out}")
    print(f"Saved data_dict_processed to: {data_processed_out}")

    print("\n" + "=" * 80)
    print("All merged pkl files saved successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()