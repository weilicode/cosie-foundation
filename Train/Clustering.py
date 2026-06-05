import os
import gc
import pickle
import joblib
import argparse
import numpy as np
import scanpy as sc
from sklearn.decomposition import IncrementalPCA

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.downstream_analysis import *



# =========================================================
# 0. Basic settings
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser("COSIE clustering pipeline")

    parser.add_argument("--project-root", type=str, required=True)
    parser.add_argument("--n-clusters", type=int, default=25)
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
    run_metacell = args.run_metacell
    
    # incremental PCA settings
    pca_dim = 50
    batch_size = 200000
    
    # Joint clustering settings
    n_clusters = args.n_clusters
    
    # =========================================================
    # 1. Define paths
    # =========================================================
    project_root = args.project_root
    
    data_root = os.path.join(project_root, "Data_preprocessing")
    embedding_root = os.path.join(project_root, "Embedding")
    clustering_root = os.path.join(project_root, "Clustering")
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data_preprocessing not found: {data_root}")
    
    os.makedirs(clustering_root, exist_ok=True)
    
    
    # Folder containing original cell-level embeddings
    embedding_folder = os.path.join(embedding_root, "final_embeddings_ori")
    
    # Folder to save PCA embeddings
    pca_embedding_folder = os.path.join(clustering_root, "final_embeddings_pca_50d")
    os.makedirs(pca_embedding_folder, exist_ok=True)
    
    # PCA model save paths
    pca_joblib_path = os.path.join(clustering_root, "joint_embedding_PCA_50d.joblib")
    pca_pkl_path = os.path.join(clustering_root, "joint_embedding_PCA_50d.pkl")
    
    # Cluster label save path
    cluster_label_pkl = os.path.join(
        clustering_root,
        f"cluster_label_{n_clusters}clusters.pkl"
    )
    
    # Folder to save clustering visualization results
    vis_clustering_root = os.path.join(clustering_root, "Vis")
    os.makedirs(vis_clustering_root, exist_ok=True)
    
    
    # =========================================================
    # 2. Load HE-only AnnData list for visualization
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Load HE-only AnnData list")
    print("=" * 80)
    
    with open(os.path.join(data_root, "HE", "data_dict_HE_only.pkl"), "rb") as f:
        data_dict_HE_only = pickle.load(f)
    
    # data_dict_HE_only is expected to be:
    # {"HE": [adata1, adata2, ...]} with spatial location saved in .obsm['spatial']
    if "HE" not in data_dict_HE_only:
        raise ValueError("data_dict_HE_only.pkl does not contain key 'HE'")
    
    adata_he_list = data_dict_HE_only["HE"]
    n_sections = len(adata_he_list)
    
    print(f"Loaded HE-only AnnData list, total sections = {n_sections}")
    
    # =========================================================
    # 3. Load final cell embeddings automatically
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Load final cell embeddings")
    print("=" * 80)

    if run_metacell:
        final_embeddings = {}
        
        for i in range(1, n_sections + 1):
            sec_key = f"s{i}"
            fname = f"{sec_key}_embedding.npy"
            fpath = os.path.join(embedding_folder, fname)
        
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing embedding file: {fpath}")
        
            print(f"Loading {fname}")
            arr = np.load(fpath)
        
            # keep float16 to reduce memory if needed
            if arr.dtype != np.float16:
                arr = arr.astype(np.float16)
        
            final_embeddings[sec_key] = arr
            print(f"  {sec_key}: shape={arr.shape}, dtype={arr.dtype}")

    else:
        with open(os.path.join(embedding_root, "final_embeddings_cell.pkl"), "rb") as f:
            final_embeddings = pickle.load(f)

        if not isinstance(final_embeddings, dict):
            raise TypeError("final_embeddings_cell.pkl should be a dict")
        
        missing_keys = [f"s{i}" for i in range(1, n_sections + 1) if f"s{i}" not in final_embeddings]
        if len(missing_keys) > 0:
            raise KeyError(f"Missing section keys in final_embeddings_cell.pkl: {missing_keys}")
    
    print("\nLoaded embeddings for all sections successfully.")
    print("Example keys:", list(final_embeddings.keys())[:5])
    print("Total sections =", len(final_embeddings))
    
    # =========================================================
    # 4. Run joint Incremental PCA (partial_fit)
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Run joint Incremental PCA (partial_fit)")
    print("=" * 80)
    
    ipca = IncrementalPCA(n_components=pca_dim)
    
    for i in range(1, n_sections + 1):
        sec_key = f"s{i}"
        emb = final_embeddings[sec_key]
    
        print(f"[{i}/{n_sections}] Fitting {sec_key}, shape={emb.shape}")
    
        for start in range(0, emb.shape[0], batch_size):
            end = min(start + batch_size, emb.shape[0])
    
            batch = emb[start:end]
    
            # partial_fit requires n_samples_in_batch >= n_components
            if batch.shape[0] >= pca_dim:
                ipca.partial_fit(batch)
            else:
                print(f"  [Warning] Skip small batch for {sec_key}: {batch.shape[0]} < pca_dim={pca_dim}")
    
            gc.collect()
    
    print("\nIncremental PCA fitting finished.")
    
    # =========================================================
    # 5. Save PCA model
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Save PCA model")
    print("=" * 80)
    
    joblib.dump(ipca, pca_joblib_path)
    print(f"Saved PCA joblib model to: {pca_joblib_path}")
    
    with open(pca_pkl_path, "wb") as f:
        pickle.dump(ipca, f)
    print(f"Saved PCA pickle model to: {pca_pkl_path}")
    
    # =========================================================
    # 6. Transform each section and save PCA embeddings
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Transform each section with fitted PCA")
    print("=" * 80)
    
    final_embeddings_pca = {}
    
    for i in range(1, n_sections + 1):
        sec_key = f"s{i}"
        emb = final_embeddings[sec_key]
    
        print(f"[{i}/{n_sections}] Transform {sec_key}, shape={emb.shape}")
    
        X_pca_list = []
        for start in range(0, emb.shape[0], batch_size):
            end = min(start + batch_size, emb.shape[0])
            X_pca_list.append(ipca.transform(emb[start:end]))
    
        X_pca = np.vstack(X_pca_list).astype(np.float32)
        final_embeddings_pca[sec_key] = X_pca
    
        save_file = os.path.join(pca_embedding_folder, f"{sec_key}_pca_embedding_50d.npy")
        np.save(save_file, X_pca)
        print(f"  Saved PCA embedding to: {save_file}, shape={X_pca.shape}, dtype={X_pca.dtype}")
    
        gc.collect()
    
    print("\nIncremental PCA transform finished.")
    
    # =========================================================
    # 7. Reload PCA embeddings from disk (optional but clean)
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Reload PCA embeddings from disk")
    print("=" * 80)
    
    final_embeddings_pca = {}
    
    for i in range(1, n_sections + 1):
        sec_key = f"s{i}"
        fpath = os.path.join(pca_embedding_folder, f"{sec_key}_pca_embedding_50d.npy")
    
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing PCA embedding file: {fpath}")
    
        arr = np.load(fpath)
        final_embeddings_pca[sec_key] = arr
    
        print(f"Loaded {sec_key} -> shape={arr.shape}, dtype={arr.dtype}")
    
    print("\nPCA embeddings loaded successfully.")
    print("Total sections =", len(final_embeddings_pca))
    print("Example keys:", list(final_embeddings_pca.keys())[:5])
    
    # =========================================================
    # 8. Define colormap
    # =========================================================
    color_map = [
        [247,182,210],[23,190,207],[44,160,44],[188,189,34],[16,60,90],
        [227,119,194],[127,127,127],[148,103,189],[214,39,40],[174,199,232],
        [255,187,120],[255,127,14],[255,152,150],[197,176,213],[196,156,148],
        [152,223,138],[199,199,199],[219,219,141],[158,218,229],[205,92,92],
        [31,119,180],[255,99,71],[46,139,87],[255,215,0],[140,86,75],
        [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
        [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0],
        [0,191,255],[255,140,0],[138,43,226],[102,205,170],[47,79,79]
    ]
    
    # =========================================================
    # 9. Joint clustering + visualization
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Joint clustering and visualization")
    print("=" * 80)
    
    cluster_label = cluster_and_visualize_superpixel(
        final_embeddings_pca,
        data_dict_HE_only,
        n_clusters=n_clusters,
        mode="joint",
        vis_basis="spatial",
        colormap=color_map,
        save_path=os.path.join(vis_clustering_root, "Clustering"), 
        dpi=300,
        figscale=150
    )
    
    print("Joint clustering finished.")
    
    # =========================================================
    # 10. Save clustering outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 8. Save clustering outputs")
    print("=" * 80)
    
    with open(cluster_label_pkl, "wb") as f:
        pickle.dump(cluster_label, f)
    
    print(f"Saved cluster labels to: {cluster_label_pkl}")
    print(f"Saved clustering figure to: {vis_clustering_root}")
    
    print("\n" + "=" * 80)
    print("All joint clustering steps completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()