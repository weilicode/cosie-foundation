#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import pickle
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad
import joblib
import torch
from sklearn.metrics import pairwise_distances_argmin

from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.downstream_analysis import cluster_and_visualize_superpixel
from COSIE_Foundation.tl import map_embedding


def parse_args():
    parser = argparse.ArgumentParser("COSIE inference pipeline")

    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output root directory. Checkpoints are expected under <out-root>/COSIE_Foundation_checkpoint/"
    )
    parser.add_argument(
        "--adata-path",
        type=str,
        required=True,
        help="Path to query h5ad file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================
    # 0. Fixed paths from args
    # =========================================================
    OUT_ROOT = Path(args.out_root)
    CKPT_ROOT = OUT_ROOT / "COSIE_Foundation_checkpoint"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    adata_path = Path(args.adata_path)
    section_key = "s1"
    out_dir = OUT_ROOT

    # =========================================================
    # 1. Global config
    # =========================================================
    config = get_default_config()
    setup_seed(config["training"]["seed"])

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # =========================================================
    # 2. Load paths
    # =========================================================
    pca_path = CKPT_ROOT / "joint_HE_PCA_model.joblib"
    ref_path = CKPT_ROOT / "adata_combined_HE.h5ad"
    feature_dict_path = CKPT_ROOT / "feature_dict_concat.pkl"
    ckpt_path = CKPT_ROOT / "cosie_trained.pt"
    centroid_path = CKPT_ROOT / "cluster_centroid.npy"
    emb_pca_path = CKPT_ROOT / "joint_embedding_PCA.joblib"

    required_files = [
        adata_path,
        pca_path,
        ref_path,
        feature_dict_path,
        ckpt_path,
        centroid_path,
        emb_pca_path,
    ]

    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    print("All required files found.")

    # =========================================================
    # 3. Read query adata and decide whether to use metacell
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 1. Read query adata and decide metacell mode")
    print("=" * 80)

    adata_query_raw = sc.read_h5ad(str(adata_path), backed="r")

    use_metacell = adata_query_raw.n_obs > 500000
    block_size = 2

    print(f"Query n_obs = {adata_query_raw.n_obs}")
    print(f"use_metacell = {use_metacell}, block_size = {block_size}")

    if use_metacell:
        adata_meta = build_metacells_grid_fast(
            adata_query_raw,
            block_size=block_size,
            spatial_key="spatial"
        )

        meta_id_per_cell = adata_meta.uns["meta_id_per_cell"]
        n_meta = adata_meta.n_obs
        print(f"Original n = {adata_query_raw.n_obs} -> Metacell n = {n_meta}")

        print("Aggregating dense X to metacell mean ...")
        X_meta_mean = aggregate_X_to_metacell_mean_dense(
            adata_query_raw.X,
            meta_id_per_cell,
            n_meta=n_meta,
            bs=20000
        )

        adata_meta.obsm["X_2048_mean"] = X_meta_mean
        adata_work = adata_meta
        n = n_meta

    else:
        meta_id_per_cell = None
        adata_work = adata_query_raw
        n = adata_work.n_obs

    # =========================================================
    # 4. HE PCA transform before mapping
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 2. Run HE PCA transform")
    print("=" * 80)

    pca = joblib.load(str(pca_path))
    k = pca.n_components_
    print(f"PCA target dim = {k}, n = {n}")

    X_pca_mm_path = out_dir / "X_pca_50d.float32.dat"
    X_pca_mm = np.memmap(X_pca_mm_path, dtype="float32", mode="w+", shape=(n, k))

    if use_metacell:
        X2048 = adata_work.obsm["X_2048_mean"]
        bs = 200000

        for s in range(0, n, bs):
            e = min(s + bs, n)
            print(f"PCA(meta) batch: {s}:{e}")
            Xb = np.asarray(X2048[s:e], dtype=np.float32, order="C")
            X_pca_mm[s:e] = pca.transform(Xb).astype(np.float32, copy=False)
            del Xb
            if (s // bs) % 10 == 0:
                gc.collect()

        del adata_work.obsm["X_2048_mean"]
        gc.collect()

    else:
        X = adata_work.X
        bs = 200000

        for s in range(0, n, bs):
            e = min(s + bs, n)
            print(f"PCA batch: {s}:{e}")

            Xb = X[s:e]
            if hasattr(Xb, "toarray"):
                Xb = Xb.toarray()
            Xb = np.asarray(Xb, dtype=np.float32, order="C")

            X_pca_mm[s:e] = pca.transform(Xb).astype(np.float32, copy=False)
            del Xb
            if (s // bs) % 10 == 0:
                gc.collect()

    X_pca_mm.flush()
    del pca
    gc.collect()

    X_pca_50d = np.asarray(X_pca_mm, dtype=np.float32)
    del X_pca_mm
    gc.collect()

    if X_pca_mm_path.exists():
        os.remove(X_pca_mm_path)

    # =========================================================
    # 5. Build adata_query for mapping + COSIE
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 3. Build query AnnData for mapping")
    print("=" * 80)

    adata_query = ad.AnnData(X=np.zeros((n, 0), dtype=np.float32))
    adata_query.obsm["spatial"] = np.asarray(adata_work.obsm["spatial"], dtype=np.float32)
    adata_query.obsm["X_pca"] = X_pca_50d

    adata_query_raw.file.close()
    del adata_query_raw
    if use_metacell:
        del adata_work
    gc.collect()

    # =========================================================
    # 6. Symphony / Harmony mapping
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 4. Run Symphony/Harmony mapping")
    print("=" * 80)

    adata_ref = sc.read_h5ad(str(ref_path), backed="r")

    map_embedding(
        adata_query=adata_query,
        adata_ref=adata_ref,
        transferred_adjusted_basis="X_pca_harmony",
        transferred_primary_basis="X_pca",
    )

    adata_ref.file.close()
    del adata_ref
    gc.collect()

    if "X_pca" in adata_query.obsm:
        del adata_query.obsm["X_pca"]
        gc.collect()

    # =========================================================
    # 7. COSIE inference
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Run COSIE inference")
    print("=" * 80)

    with open(feature_dict_path, "rb") as f:
        feature_dict_train = pickle.load(f)

    model = COSIE_model(config, feature_dict_train)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))

    del feature_dict_train
    gc.collect()

    HE_dict = {"HE": torch.from_numpy(adata_query.obsm["X_pca_harmony"]).float()}
    feature_dict_test = {section_key: HE_dict}
    spatial_loc_dict_test = {section_key: adata_query.obsm["spatial"]}

    final_embeddings_test = infer_embeddings(
        model,
        feature_dict_test,
        spatial_loc_dict_test,
        device,
        config["training"]["knn_neighbors_spatial"],
        config["training"]["knn_neighbors_feature"],
    )

    cosie_emb_meta = final_embeddings_test[section_key]

    del model, final_embeddings_test, HE_dict, feature_dict_test, spatial_loc_dict_test
    gc.collect()

    # =========================================================
    # 8. Embedding PCA + label transfer
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Run embedding PCA + label transfer")
    print("=" * 80)

    C = np.load(str(centroid_path))
    pca_emb = joblib.load(str(emb_pca_path))

    embedding_query_pca_meta = pca_emb.transform(cosie_emb_meta)
    assigned_idx = pairwise_distances_argmin(embedding_query_pca_meta, C)
    assigned_labels_meta = np.arange(C.shape[0])[assigned_idx].astype(np.int32)

    del C, pca_emb, assigned_idx
    gc.collect()

    # =========================================================
    # 9. Save output AnnData
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Save output AnnData")
    print("=" * 80)

    if use_metacell:
        cosie_emb_cell = broadcast_back(meta_id_per_cell, cosie_emb_meta).astype(np.float32, copy=False)
        X_ipca_cell = broadcast_back(meta_id_per_cell, embedding_query_pca_meta).astype(np.float32, copy=False)
        X_symphony_cell = broadcast_back(meta_id_per_cell, adata_query.obsm["X_pca_harmony"]).astype(np.float32, copy=False)
        X_input_pca_cell = broadcast_back(meta_id_per_cell, X_pca_50d).astype(np.float32, copy=False)
        assigned_labels_cell = assigned_labels_meta[meta_id_per_cell].astype(np.int32, copy=False)

        adata_raw_for_output = sc.read_h5ad(str(adata_path), backed="r")
        spatial_cell = np.asarray(adata_raw_for_output.obsm["spatial"]).copy()
        adata_raw_for_output.file.close()
        del adata_raw_for_output
        gc.collect()

        adata_out = ad.AnnData(X=np.zeros((spatial_cell.shape[0], 0), dtype=np.float32))
        adata_out.obsm["spatial"] = spatial_cell.astype(np.float32, copy=False)
        adata_out.obsm["X_cosie"] = cosie_emb_cell
        adata_out.obsm["X_ipca"] = X_ipca_cell
        adata_out.obsm["X_symphony"] = X_symphony_cell
        adata_out.obsm["X_input_pca"] = X_input_pca_cell
        adata_out.obs["assigned_label"] = assigned_labels_cell

        adata_meta_out = ad.AnnData(X=np.zeros((adata_query.n_obs, 0), dtype=np.float32))
        adata_meta_out.obsm["spatial"] = np.asarray(adata_query.obsm["spatial"], dtype=np.float32)
        adata_meta_out.obsm["X_cosie"] = cosie_emb_meta.astype(np.float32, copy=False)
        adata_meta_out.obsm["X_ipca"] = embedding_query_pca_meta.astype(np.float32, copy=False)
        adata_meta_out.obs["assigned_label"] = assigned_labels_meta.astype(np.int32, copy=False)
        adata_meta_out.uns["meta_id_per_cell"] = meta_id_per_cell.astype(np.int32, copy=False)
        adata_meta_out.uns["original_n_obs"] = int(spatial_cell.shape[0])
        adata_meta_out.uns["block_size"] = int(block_size)

        save_meta_h5ad_path = out_dir / "adata_query_metacell_inferred.h5ad"
        adata_meta_out.write(save_meta_h5ad_path)
        print(f"Saved metacell AnnData to: {save_meta_h5ad_path}")

    else:
        adata_out = ad.AnnData(X=np.zeros((adata_query.n_obs, 0), dtype=np.float32))
        adata_out.obsm["spatial"] = np.asarray(adata_query.obsm["spatial"], dtype=np.float32)
        adata_out.obsm["X_cosie"] = cosie_emb_meta.astype(np.float32, copy=False)
        adata_out.obsm["X_ipca"] = embedding_query_pca_meta.astype(np.float32, copy=False)
        adata_out.obsm["X_symphony"] = adata_query.obsm["X_pca_harmony"].astype(np.float32, copy=False)
        adata_out.obsm["X_input_pca"] = X_pca_50d.astype(np.float32, copy=False)
        adata_out.obs["assigned_label"] = assigned_labels_meta.astype(np.int32, copy=False)

    # =========================================================
    # 10. Convert transferred cluster labels to cell-type labels
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 8. Convert cluster labels to cell-type labels")
    print("=" * 80)

    cluster_key = "assigned_label"
    group_key = "celltype_labels"

    group_dict = {
        "Macrophages": [11],
        "Bronchus": [3],
        "Vessels": [8, 10],
        "Normal lung": [1, 4, 9, 13, 15, 18, 21, 23],
        "Pneumocytes": [17],
        "Tumor": [2, 5, 6, 7, 14, 16, 19],
        "Fibrous tissue": [0, 12, 22],
        "Lymphoid aggregates": [24],
        "Fibrous+tumor": [20],
    }

    colormap = [
        [255, 127, 14],   # Macrophages
        [188, 189, 34],   # Bronchus
        [148, 103, 189],  # Vessels
        [173, 216, 230],  # Normal lung
        [77, 175, 74],    # Pneumocytes
        [220, 20, 60],    # Tumor
        [247, 182, 210],  # Fibrous tissue
        [139, 69, 19],    # Lymphoid aggregates
        [0, 191, 255],    # Fibrous+tumor
    ]

    legend_labels = [
        f"{name} (clusters {','.join(map(str, clusters))})"
        for name, clusters in group_dict.items()
    ]

    group_labels = assign_group_from_clusters(
        adata_out,
        cluster_key=cluster_key,
        group_dict=group_dict,
        new_key=group_key,
    )

    save_h5ad_path = out_dir / "adata_query_inferred.h5ad"
    adata_out.write(save_h5ad_path)
    print(f"Saved final inferred AnnData to: {save_h5ad_path}")

    # =========================================================
    # 11. Save only ONE final cell-level plot
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 9. Save final cell-level cell-type plot")
    print("=" * 80)

    fig_path = out_dir / "celltype_labels.png"

    visualize_superpixel_from_adata(
        adata_out,
        obs_key=group_key,
        colormap=colormap,
        legend_labels=legend_labels,
        swap_xy=True,
        figscale=200,
        save_path=fig_path,
    )

    print(f"Saved final cell-level plot to: {fig_path}")

    # =========================================================
    # 12. Cleanup
    # =========================================================
    del adata_query, adata_out, cosie_emb_meta, embedding_query_pca_meta, assigned_labels_meta
    gc.collect()

    print("\n" + "=" * 80)
    print("All inference outputs saved to:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()