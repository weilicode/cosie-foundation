import os
import gc
import pickle
import joblib
import argparse
import numpy as np
from sklearn.decomposition import IncrementalPCA

from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.downstream_analysis import *


def parse_args():
    parser = argparse.ArgumentParser("COSIE clustering pipeline")
    parser.add_argument("--project-root", type=str, required=True)
    parser.add_argument("--n-clusters", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================
    # 0. Basic settings
    # =========================================================
    config = get_default_config()
    setup_seed(config["training"]["seed"])

    pca_dim = 50
    batch_size = 200000
    n_clusters = args.n_clusters

    # =========================================================
    # 1. Paths
    # =========================================================
    project_root = args.project_root
    data_root = os.path.join(project_root, "Data_preprocessing")
    embedding_root = os.path.join(project_root, "Embedding")
    clustering_root = os.path.join(project_root, "Clustering")

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data_preprocessing not found: {data_root}")

    os.makedirs(clustering_root, exist_ok=True)

    embedding_pkl = os.path.join(embedding_root, "final_embeddings_cell.pkl")
    pca_embedding_folder = os.path.join(clustering_root, "final_embeddings_pca_50d")
    os.makedirs(pca_embedding_folder, exist_ok=True)

    pca_joblib_path = os.path.join(clustering_root, "joint_embedding_PCA_50d.joblib")
    pca_pkl_path = os.path.join(clustering_root, "joint_embedding_PCA_50d.pkl")
    cluster_label_pkl = os.path.join(clustering_root, f"cluster_label_{n_clusters}clusters.pkl")

    vis_clustering_root = os.path.join(clustering_root, "Vis")
    os.makedirs(vis_clustering_root, exist_ok=True)

    # =========================================================
    # 2. Load HE-only AnnData
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Load HE-only AnnData list")
    print("=" * 80)

    he_pkl = os.path.join(data_root, "HE", "data_dict_HE_only.pkl")
    with open(he_pkl, "rb") as f:
        data_dict_HE_only = pickle.load(f)

    if "HE" not in data_dict_HE_only:
        raise ValueError("data_dict_HE_only.pkl does not contain key 'HE'")

    adata_he_list = data_dict_HE_only["HE"]
    n_sections = len(adata_he_list)

    print(f"Loaded HE-only AnnData list, total sections = {n_sections}")

    # =========================================================
    # 3. Load final embeddings
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Load final embeddings")
    print("=" * 80)

    if not os.path.exists(embedding_pkl):
        raise FileNotFoundError(f"Missing embedding file: {embedding_pkl}")

    with open(embedding_pkl, "rb") as f:
        final_embeddings = pickle.load(f)

    if not isinstance(final_embeddings, dict):
        raise TypeError("final_embeddings_cell.pkl should be a dict")

    expected_keys = [f"s{i}" for i in range(1, n_sections + 1)]
    missing_keys = [k for k in expected_keys if k not in final_embeddings]

    if missing_keys:
        raise KeyError(f"Missing section keys in final_embeddings_cell.pkl: {missing_keys}")

    print("Loaded embeddings for all sections successfully.")
    print("Example keys:", list(final_embeddings.keys())[:5])
    print("Total sections =", len(final_embeddings))

    # =========================================================
    # 4. Fit joint Incremental PCA
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Run joint Incremental PCA")
    print("=" * 80)

    ipca = IncrementalPCA(n_components=pca_dim)

    for i, sec_key in enumerate(expected_keys, start=1):
        emb = final_embeddings[sec_key]
        print(f"[{i}/{n_sections}] Fitting {sec_key}, shape={emb.shape}")

        for start in range(0, emb.shape[0], batch_size):
            end = min(start + batch_size, emb.shape[0])
            batch = emb[start:end]

            if batch.shape[0] >= pca_dim:
                ipca.partial_fit(batch)
            else:
                print(
                    f"  [Warning] Skip small batch for {sec_key}: "
                    f"{batch.shape[0]} < pca_dim={pca_dim}"
                )

        gc.collect()

    print("\nIncremental PCA fitting finished.")

    # =========================================================
    # 5. Save PCA model
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Save PCA model")
    print("=" * 80)

    joblib.dump(ipca, pca_joblib_path)

    with open(pca_pkl_path, "wb") as f:
        pickle.dump(ipca, f)

    print(f"Saved PCA joblib model to: {pca_joblib_path}")
    print(f"Saved PCA pickle model to: {pca_pkl_path}")

    # =========================================================
    # 6. Transform each section
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Transform each section with fitted PCA")
    print("=" * 80)

    final_embeddings_pca = {}

    for i, sec_key in enumerate(expected_keys, start=1):
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

        print(f"  Saved PCA embedding to: {save_file}, shape={X_pca.shape}")
        gc.collect()

    print("\nIncremental PCA transform finished.")

    # =========================================================
    # 7. Colormap
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
    # 8. Joint clustering
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Joint clustering and visualization")
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
    # 9. Save outputs
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Save clustering outputs")
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