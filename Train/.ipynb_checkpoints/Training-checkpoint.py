import os
import gc
import pickle
import argparse
import numpy as np
import torch
import scanpy as sc

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.downstream_analysis import *



def parse_args():
    parser = argparse.ArgumentParser("COSIE training pipeline")

    parser.add_argument("--project-root", type=str, required=True)

    parser.add_argument(
        "--run-metacell",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
        help="Whether to recover original embeddings (default: True)"
    )

    return parser.parse_args()

def main():
    
    # =========================================================
    # 0. Basic settings
    # =========================================================
    args = parse_args()
    
    config = get_default_config()
    setup_seed(config["training"]["seed"])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # =========================================================
    # 1. Define paths
    # =========================================================
    project_root = args.project_root
    data_root = os.path.join(project_root, "Data_preprocessing")
    training_root = os.path.join(project_root, "Training")
    embedding_root = os.path.join(project_root, "Embedding")
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data_preprocessing not found: {data_root}")
    
    run_metacell = args.run_metacell
    
    os.makedirs(training_root, exist_ok=True)
    os.makedirs(embedding_root, exist_ok=True)
        
    if run_metacell:
        he_meta_root = os.path.join(data_root, "HE", "meta_cell_2_2")
        final_embedding_folder = os.path.join(embedding_root, "final_embeddings_ori")
        os.makedirs(final_embedding_folder, exist_ok=True)
    
    # Checkpoint path
    checkpoint_path = os.path.join(training_root, "cosie_trained.pt")
    
    # Save subset-related files
    subset_index_path = os.path.join(training_root, "sub_indices_list.pkl")
    subset_label_path = os.path.join(training_root, "labels_list.pkl")
    subset_dict_path = os.path.join(training_root, "sub_indices_dict.pkl")
    linkage_path = os.path.join(training_root, "Linkage_indicator.pkl")
    group_info_path = os.path.join(training_root, "section_group_info.pkl")
    
    # Final metacell embedding save
    cell_embedding_pkl = os.path.join(embedding_root, "final_embeddings_cell.pkl")
    
    
    # =========================================================
    # 2. Load merged preprocessing outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Load merged preprocessing outputs")
    print("=" * 80)
    
    with open(os.path.join(data_root, "feature_dict_concat.pkl"), "rb") as f:
        feature_dict = pickle.load(f)
    
    with open(os.path.join(data_root, "data_dict_processed_concat.pkl"), "rb") as f:
        data_dict_processed = pickle.load(f)
    
    # spatial_loc_dict is still needed for training/inference
    # Use HE spatial location dict as the shared spatial reference
    with open(os.path.join(data_root, "spatial_loc_dict.pkl"), "rb") as f:
        spatial_loc_dict = pickle.load(f)
    
    print("Loaded feature_dict sections:", len(feature_dict))
    print("Loaded data_dict_processed keys:", data_dict_processed.keys())
    print("Loaded spatial_loc_dict sections:", len(spatial_loc_dict))
    
    # =========================================================
    # 3. Automatically determine section list
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Determine section list automatically")
    print("=" * 80)
    
    section_keys = sorted(feature_dict.keys(), key=lambda x: int(x[1:]))  # s1, s2, ...
    n_sections = len(section_keys)
    
    print(f"Total sections = {n_sections}")
    print("Section keys:", section_keys)
    
    # Also check HE list length
    if "HE" not in data_dict_processed:
        raise ValueError("data_dict_processed does not contain 'HE'.")
    
    if len(data_dict_processed["HE"]) != n_sections:
        print("[Warning] len(data_dict_processed['HE']) != n_sections")
        print("  len(data_dict_processed['HE']) =", len(data_dict_processed["HE"]))
        print("  n_sections =", n_sections)
    
    # =========================================================
    # 4. Build subset based on HE
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Build subset indices using HE")
    print("=" * 80)
    
    sub_indices_list = []
    labels_list = []
    
    for i in range(n_sections):
        sec_key = f"s{i+1}"
        print(f"Processing {sec_key}")
    
        adata_he = data_dict_processed["HE"][i]
        if adata_he is None:
            raise ValueError(f"HE data is None for {sec_key}. Cannot build subset from HE.")
    
        X = adata_he.X
    
        sub_idx, labels, _ = subsample_by_kmeans(
            X=X,
            cluster_num=25,
            sample_ratio_each_cluster=0.05
        )
    
        sub_indices_list.append(sub_idx)
        labels_list.append(labels)
    
    with open(subset_index_path, "wb") as f:
        pickle.dump(sub_indices_list, f)
    
    with open(subset_label_path, "wb") as f:
        pickle.dump(labels_list, f)
    
    print(f"Saved sub_indices_list to: {subset_index_path}")
    print(f"Saved labels_list to: {subset_label_path}")
    
    # Create section-keyed subset dict
    sub_indices_dict = {
        f"s{i+1}": sub_indices_list[i]
        for i in range(n_sections)
    }
    
    with open(subset_dict_path, "wb") as f:
        pickle.dump(sub_indices_dict, f)
    
    print(f"Saved sub_indices_dict to: {subset_dict_path}")
    
    # Subset COSIE inputs
    feature_dict_sub, spatial_loc_dict_sub, data_dict_processed_sub = subset_cosie_inputs(
        feature_dict,
        spatial_loc_dict,
        data_dict_processed,
        sub_indices_dict
    )
    
    print("Subset completed.")
    print("Subset feature_dict sections:", len(feature_dict_sub))
    
    # =========================================================
    # 5. Compute entropy for representative selection
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Compute entropy for each section")
    print("=" * 80)
    
    entropies = [cluster_entropy(l) for l in labels_list]
    
    for i, e in enumerate(entropies, start=1):
        print(f"s{i}: entropy = {e:.4f}")
    
    # =========================================================
    # 6. Automatically group sections by modality combination
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Group sections by modality combination")
    print("=" * 80)
    
    # Example:
    #   ('HE', 'RNA')
    #   ('HE', 'RNA', 'Protein')
    group_dict = {}
    
    for sec in section_keys:
        mods = tuple(sorted(feature_dict[sec].keys()))
        if mods not in group_dict:
            group_dict[mods] = []
        group_dict[mods].append(sec)
    
    print("Section groups by modality combination:")
    for mods, secs in group_dict.items():
        print(f"  {mods}: {secs}")
    
    # Save group info
    with open(group_info_path, "wb") as f:
        pickle.dump(group_dict, f)
    
    print(f"Saved group info to: {group_info_path}")
    
    # =========================================================
    # 7. Select one representative section per group
    #    Representative = section with maximum entropy in that group
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Select representative section for each group")
    print("=" * 80)
    
    rep_dict = {}
    
    for mods, secs in group_dict.items():
        best_sec = None
        best_entropy = -np.inf
    
        for sec in secs:
            idx = int(sec[1:]) - 1
            e = entropies[idx]
            if e > best_entropy:
                best_entropy = e
                best_sec = sec
    
        rep_dict[mods] = best_sec
        print(f"Group {mods} -> representative {best_sec} (entropy = {best_entropy:.4f})")
    
    # =========================================================
    # 8. Build Linkage_indicator automatically
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Build Linkage_indicator automatically")
    print("=" * 80)
    
    Linkage_indicator = {}
    
    # -------------------------
    # 8.1 Within-group linkages
    # -------------------------
    # For sections in the same modality-combination group:
    # link common modalities to themselves
    for mods, secs in group_dict.items():
        rep_sec = rep_dict[mods]
    
        for sec in secs:
            if sec == rep_sec:
                continue
    
            linkage_pairs = [(m, m) for m in mods]
            Linkage_indicator[(rep_sec, sec)] = linkage_pairs
    
    # -------------------------
    # 8.2 Cross-group linkages
    # -------------------------
    # For two different groups:
    # link representative sections by shared modalities with same-name pairs
    # and optionally add cross-modality RNA<->Protein if both exist
    group_mods_list = list(group_dict.keys())
    
    for i in range(len(group_mods_list)):
        for j in range(i + 1, len(group_mods_list)):
            mods_a = group_mods_list[i]
            mods_b = group_mods_list[j]
    
            rep_a = rep_dict[mods_a]
            rep_b = rep_dict[mods_b]
    
            linkage_pairs = []
    
            # shared modalities: HE-HE, RNA-RNA, Protein-Protein
            shared_mods = set(mods_a).intersection(set(mods_b))
            for m in sorted(shared_mods):
                linkage_pairs.append((m, m))
    
            # optional cross-modality linkage:
            # if one side has RNA and the other has Protein
            if ("RNA" in mods_a) and ("Protein" in mods_b):
                linkage_pairs.append(("RNA", "Protein"))
    
            if ("Protein" in mods_a) and ("RNA" in mods_b):
                linkage_pairs.append(("Protein", "RNA"))
    
            # only create linkage if there is at least one pair
            if len(linkage_pairs) > 0:
                Linkage_indicator[(rep_a, rep_b)] = linkage_pairs
                Linkage_indicator[(rep_b, rep_a)] = [(b, a) for (a, b) in linkage_pairs]
    
    print("Constructed Linkage_indicator:")
    for k, v in Linkage_indicator.items():
        print(f"  {k}: {v}")
    
    with open(linkage_path, "wb") as f:
        pickle.dump(Linkage_indicator, f)
    
    print(f"Saved Linkage_indicator to: {linkage_path}")
    
    # =========================================================
    # 9. Train model on subset data
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 8. Train COSIE model on subset data")
    print("=" * 80)
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    model = COSIE_model(config, feature_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    
    print(f"Training output directory: {training_root}")
    print(f"Checkpoint will be saved to: {checkpoint_path}")
    
    
    final_embeddings_subset = model.train_model(
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
    
    print("Subset training finished.")
    
    
    # =========================================================
    # 10. Rebuild model and load checkpoint for full inference
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 9. Reload trained checkpoint for full inference")
    print("=" * 80)
    
    torch.cuda.empty_cache()
    
    model = COSIE_model(config, feature_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    
    if not os.path.exists(checkpoint_path):
        # Fallback: try to detect a .pt file in training_root
        pt_files = [x for x in os.listdir(training_root) if x.endswith(".pt")]
        if len(pt_files) == 1:
            checkpoint_path = os.path.join(training_root, pt_files[0])
            print(f"[Warning] Using detected checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}, and could not uniquely infer one from {training_root}"
            )
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # =========================================================
    # 11. Perform inference on all metacells
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 10. Infer cell embeddings on full data")
    print("=" * 80)
    
    final_embeddings_mc = infer_embeddings(
        model,
        feature_dict,
        spatial_loc_dict,
        device,
        config["training"]["knn_neighbors_spatial"],
        config["training"]["knn_neighbors_feature"]
    )
    
    with open(cell_embedding_pkl, "wb") as f:
        pickle.dump(final_embeddings_mc, f)
    
    print(f"Saved cell embeddings to: {cell_embedding_pkl}")
    
    torch.cuda.empty_cache()
    
    if run_metacell:
        # =========================================================
        # 12. Load HE meta-to-original mappings
        # =========================================================
        print("\n" + "=" * 80)
        print("Step 11. Load HE meta-to-original mappings")
        print("=" * 80)
        
        meta_he_maps = []
        
        for i in range(1, n_sections + 1):
            pkl_path = os.path.join(he_meta_root, f"meta_to_original_s{i}.pkl")
        
            if os.path.exists(pkl_path):
                print(f"Loading meta_to_original_he_{i}.pkl")
                with open(pkl_path, "rb") as f:
                    meta_he_maps.append(pickle.load(f))
            else:
                raise FileNotFoundError(f"Missing meta mapping file: {pkl_path}")
        
        print(f"Loaded {len(meta_he_maps)} HE meta mappings")
        
        # =========================================================
        # 13. Recover original cell-level embeddings
        # =========================================================
        print("\n" + "=" * 80)
        print("Step 12. Recover original cell-level embeddings")
        print("=" * 80)
        
        for i in range(1, n_sections + 1):
            key = f"s{i}"
            print(f"Recovering {key} ...")
        
            meta_map = meta_he_maps[i - 1]
            adata_metacell = data_dict_processed["HE"][i - 1]
            metacell_emb = final_embeddings_mc[key]
        
            original_emb = reconstruct_metacell_to_original_new(
                meta_map=meta_map,
                adata_metacell=adata_metacell,
                metacell_embedding=metacell_emb
            )
        
            save_file = os.path.join(final_embedding_folder, f"{key}_embedding.npy")
            np.save(save_file, original_emb)
            print(f"Saved original embedding to: {save_file}, shape={original_emb.shape}")
        
    print("\n" + "=" * 80)
    print("All training and inference steps completed successfully.")
    print("=" * 80)
        
if __name__ == "__main__":
    main()


# ###