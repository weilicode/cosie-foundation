import os
import pickle
import argparse
import numpy as np
import torch

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.downstream_analysis import *


def parse_args():
    parser = argparse.ArgumentParser("COSIE training pipeline")
    parser.add_argument("--project-root", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================
    # 0. Basic settings
    # =========================================================
    config = get_default_config()
    setup_seed(config["training"]["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================
    # 1. Paths
    # =========================================================
    project_root = args.project_root
    data_root = os.path.join(project_root, "Data_preprocessing")
    training_root = os.path.join(project_root, "Training")

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data_preprocessing not found: {data_root}")

    os.makedirs(training_root, exist_ok=True)

    checkpoint_path = os.path.join(training_root, "cosie_trained.pt")
    subset_index_path = os.path.join(training_root, "sub_indices_list.pkl")
    subset_label_path = os.path.join(training_root, "labels_list.pkl")
    subset_dict_path = os.path.join(training_root, "sub_indices_dict.pkl")
    linkage_path = os.path.join(training_root, "Linkage_indicator.pkl")
    group_info_path = os.path.join(training_root, "section_group_info.pkl")

    # =========================================================
    # 2. Load preprocessing outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Load preprocessing outputs")
    print("=" * 80)

    with open(os.path.join(data_root, "feature_dict_concat.pkl"), "rb") as f:
        feature_dict = pickle.load(f)

    with open(os.path.join(data_root, "data_dict_processed_concat.pkl"), "rb") as f:
        data_dict_processed = pickle.load(f)

    with open(os.path.join(data_root, "spatial_loc_dict.pkl"), "rb") as f:
        spatial_loc_dict = pickle.load(f)

    print("Loaded feature_dict sections:", len(feature_dict))
    print("Loaded data_dict_processed keys:", data_dict_processed.keys())
    print("Loaded spatial_loc_dict sections:", len(spatial_loc_dict))



    section_keys = sorted(feature_dict.keys(), key=lambda x: int(x[1:]))
    n_sections = len(section_keys)


    if "HE" not in data_dict_processed:
        raise ValueError("data_dict_processed does not contain 'HE'.")

    if len(data_dict_processed["HE"]) != n_sections:
        raise ValueError(
            f"HE section number mismatch: "
            f"{len(data_dict_processed['HE'])} != {n_sections}"
        )


    sub_indices_list, labels_list = [], []

    for i, sec_key in enumerate(section_keys):
        print(f"Processing {sec_key}")

        adata_he = data_dict_processed["HE"][i]
        if adata_he is None:
            raise ValueError(f"HE data is None for {sec_key}. Cannot build subset from HE.")

        sub_idx, labels, _ = subsample_by_kmeans(
            X=adata_he.X,
            cluster_num=25,
            sample_ratio_each_cluster=0.05
        )

        sub_indices_list.append(sub_idx)
        labels_list.append(labels)

    with open(subset_index_path, "wb") as f:
        pickle.dump(sub_indices_list, f)

    with open(subset_label_path, "wb") as f:
        pickle.dump(labels_list, f)

    sub_indices_dict = {
        section_keys[i]: sub_indices_list[i]
        for i in range(n_sections)
    }

    with open(subset_dict_path, "wb") as f:
        pickle.dump(sub_indices_dict, f)

    print(f"Saved sub_indices_list to: {subset_index_path}")
    print(f"Saved labels_list to: {subset_label_path}")
    print(f"Saved sub_indices_dict to: {subset_dict_path}")

    feature_dict_sub, spatial_loc_dict_sub, data_dict_processed_sub = subset_cosie_inputs(
        feature_dict,
        spatial_loc_dict,
        data_dict_processed,
        sub_indices_dict
    )



    entropies = [cluster_entropy(x) for x in labels_list]
    entropy_dict = {sec: entropies[i] for i, sec in enumerate(section_keys)}

    for sec in section_keys:
        print(f"{sec}: entropy = {entropy_dict[sec]:.4f}")


    group_dict = {}

    for sec in section_keys:
        mods = tuple(sorted(feature_dict[sec].keys()))
        group_dict.setdefault(mods, []).append(sec)

    for mods, secs in group_dict.items():
        print(f"  {mods}: {secs}")

    with open(group_info_path, "wb") as f:
        pickle.dump(group_dict, f)



    rep_dict = {}

    for mods, secs in group_dict.items():
        best_sec = max(secs, key=lambda sec: entropy_dict[sec])
        rep_dict[mods] = best_sec
        print(
            f"Group {mods} -> representative {best_sec} "
            f"(entropy = {entropy_dict[best_sec]:.4f})"
        )


    Linkage_indicator = {}

    # Within-group linkages
    for mods, secs in group_dict.items():
        rep_sec = rep_dict[mods]

        for sec in secs:
            if sec == rep_sec:
                continue
            Linkage_indicator[(rep_sec, sec)] = [(m, m) for m in mods]

    # Cross-group linkages
    group_mods_list = list(group_dict.keys())

    for i in range(len(group_mods_list)):
        for j in range(i + 1, len(group_mods_list)):
            mods_a, mods_b = group_mods_list[i], group_mods_list[j]
            rep_a, rep_b = rep_dict[mods_a], rep_dict[mods_b]

            linkage_pairs = [
                (m, m)
                for m in sorted(set(mods_a).intersection(mods_b))
            ]

            if "RNA" in mods_a and "Protein" in mods_b:
                linkage_pairs.append(("RNA", "Protein"))

            if "Protein" in mods_a and "RNA" in mods_b:
                linkage_pairs.append(("Protein", "RNA"))

            if linkage_pairs:
                Linkage_indicator[(rep_a, rep_b)] = linkage_pairs
                Linkage_indicator[(rep_b, rep_a)] = [
                    (b, a) for a, b in linkage_pairs
                ]


    with open(linkage_path, "wb") as f:
        pickle.dump(Linkage_indicator, f)

    print(f"Saved Linkage_indicator to: {linkage_path}")

    print("\n" + "=" * 80)
    print("Train COSIE model on subset data")
    print("=" * 80)

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = COSIE_model(config, feature_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    print(f"Training output directory: {training_root}")
    print(f"Checkpoint will be saved to: {checkpoint_path}")

    model.train_model(
        training_root,
        config,
        optimizer,
        device,
        feature_dict_sub,
        spatial_loc_dict_sub,
        data_dict_processed_sub,
        Linkage_indicator,
        n_x=1,
        n_y=1
    )

    print("Training finished.")

    

if __name__ == "__main__":
    main()