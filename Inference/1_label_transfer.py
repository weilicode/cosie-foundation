#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, gc, pickle, argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import scanpy as sc
import anndata as ad
import joblib
import torch
from sklearn.metrics import pairwise_distances_argmin

from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.tl import map_embedding
from COSIE_Foundation.data_preprocessing import clr_normalize_each_cell
import symphonypy as sp

def parse_args():
    parser = argparse.ArgumentParser("COSIE inference pipeline")
    parser.add_argument("--out-root", type=str, required=True, help="Output root. Checkpoints expected under <out-root>/COSIE_Foundation_checkpoint/")
    parser.add_argument("--adata-path", type=str, required=True, help="Path to query h5ad")
    parser.add_argument("--modality", type=str, required=True, choices=["HE", "RNA", "Protein"], help="Input modality: HE, RNA, or Protein")
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================
    # 0. Paths and config
    # =========================================================
    OUT_ROOT = Path(args.out_root)
    CKPT_ROOT = OUT_ROOT / "COSIE_Foundation_checkpoint"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    adata_path = Path(args.adata_path)
    modality = args.modality
    section_key = "s1"
    out_dir = OUT_ROOT

    config = get_default_config()
    setup_seed(config["training"]["seed"])
    device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Input modality: {modality}")

    # =========================================================
    # 1. Checkpoint paths
    # =========================================================
    pca_path = CKPT_ROOT / "joint_HE_PCA_model.joblib"
    ref_path_dict = {
        "HE": CKPT_ROOT / "adata_combined_HE.h5ad",
        "RNA": CKPT_ROOT / "adata_combined_RNA.h5ad",
        "Protein": CKPT_ROOT / "adata_combined_Protein.h5ad",
    }

    ref_path = ref_path_dict[modality]
    feature_dict_path = CKPT_ROOT / "feature_dict_concat.pkl"
    ckpt_path = CKPT_ROOT / "cosie_trained.pt"
    centroid_path = CKPT_ROOT / "cluster_centroid.npy"
    emb_pca_path = CKPT_ROOT / "joint_embedding_PCA.joblib"

    required_files = [adata_path, ref_path, feature_dict_path, ckpt_path, centroid_path, emb_pca_path]
    if modality == "HE":
        required_files.append(pca_path)

    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    print("All required files found.")

    # =========================================================
    # 2. Prepare query
    # =========================================================
    print("\n" + "=" * 80)
    print(f"Step 1. Prepare {modality} query")
    print("=" * 80)

    # ---------------------------------------------------------
    # HE
    # ---------------------------------------------------------
    if modality == "HE":
        adata_query_raw = sc.read_h5ad(str(adata_path), backed="r")
        use_metacell = adata_query_raw.n_obs > 5000000
        block_size = 2

        print(f"Query n_obs = {adata_query_raw.n_obs}")

        if use_metacell:
            adata_meta = build_metacells_grid_fast(adata_query_raw, block_size=block_size, spatial_key="spatial")
            meta_id_per_cell = adata_meta.uns["meta_id_per_cell"]
            n = adata_meta.n_obs

            X_meta_mean = aggregate_X_to_metacell_mean_dense(adata_query_raw.X, meta_id_per_cell, n_meta=n, bs=20000)
            adata_meta.obsm["X_2048_mean"] = X_meta_mean
            adata_work = adata_meta
            print(f"Original n = {adata_query_raw.n_obs} -> Metacell n = {n}")
        else:
            meta_id_per_cell = None
            adata_work = adata_query_raw
            n = adata_work.n_obs

        print("\n" + "=" * 80)
        print("Step 2. Run HE PCA transform")
        print("=" * 80)

        pca = joblib.load(str(pca_path))
        k = pca.n_components_
        X_pca_mm_path = out_dir / "X_pca_50d.float32.dat"
        X_pca_mm = np.memmap(X_pca_mm_path, dtype="float32", mode="w+", shape=(n, k))
        bs = 200000

        if use_metacell:
            X2048 = adata_work.obsm["X_2048_mean"]
            for s in range(0, n, bs):
                e = min(s + bs, n)
                Xb = np.asarray(X2048[s:e], dtype=np.float32, order="C")
                X_pca_mm[s:e] = pca.transform(Xb).astype(np.float32, copy=False)
                del Xb
                if (s // bs) % 10 == 0:
                    gc.collect()
            del adata_work.obsm["X_2048_mean"]
        else:
            X = adata_work.X
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
        X_pca_50d = np.asarray(X_pca_mm, dtype=np.float32).copy()
        del X_pca_mm, pca
        gc.collect()

        if X_pca_mm_path.exists():
            os.remove(X_pca_mm_path)

        print("\n" + "=" * 80)
        print("Step 3. Build HE query AnnData")
        print("=" * 80)

        adata_query = ad.AnnData(X=np.zeros((n, 0), dtype=np.float32))
        adata_query.obsm["spatial"] = np.asarray(adata_work.obsm["spatial"], dtype=np.float32)
        adata_query.obsm["X_pca"] = X_pca_50d

        adata_query_raw.file.close()
        del adata_query_raw, adata_work, X_pca_50d
        gc.collect()

    # ---------------------------------------------------------
    # RNA / Protein
    # ---------------------------------------------------------
    else:
        use_metacell = False
        meta_id_per_cell = None
        block_size = 2

        adata_query = sc.read_h5ad(str(adata_path))
        adata_query.var_names_make_unique()
        n = adata_query.n_obs

        print(f"Query n_obs = {n}")

        if modality == "RNA":
            sc.pp.normalize_total(adata_query)
            sc.pp.log1p(adata_query)

        elif modality == "Protein":
            adata_query = clr_normalize_each_cell(adata_query)

    # =========================================================
    # 3. Symphony / Harmony mapping
    # =========================================================
    print("\n" + "=" * 80)
    print(f"Step 4. Run {modality} mapping")
    print("=" * 80)
    
    adata_ref = sc.read_h5ad(str(ref_path), backed="r")
    
    if modality == "HE":
        map_embedding(
            adata_query=adata_query,
            adata_ref=adata_ref,
            transferred_adjusted_basis="X_pca_harmony",
            transferred_primary_basis="X_pca",
        )
        input_pca_key = "X_pca"
    
    else:
        adata_ref.var["highly_variable"] = True
    
        sp.tl.map_embedding(
            adata_query=adata_query,
            adata_ref=adata_ref,
            transferred_adjusted_basis="X_pca_harmony",
            transferred_primary_basis="X_pca_reference",
        )
        input_pca_key = "X_pca_reference"
    
    X_input_pca = np.asarray(adata_query.obsm[input_pca_key], dtype=np.float32).copy()
    
    adata_ref.file.close()
    del adata_ref
    
    if input_pca_key in adata_query.obsm:
        del adata_query.obsm[input_pca_key]
    
    if "X_pca_harmony_symphony_R" in adata_query.obsm:
        del adata_query.obsm["X_pca_harmony_symphony_R"]
    
    gc.collect()

    # =========================================================
    # 4. COSIE inference
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 5. Run COSIE inference")
    print("=" * 80)

    with open(feature_dict_path, "rb") as f:
        feature_dict_train = pickle.load(f)

    model = COSIE_model(config, feature_dict_train)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    del feature_dict_train
    gc.collect()

    input_dict = {modality: torch.from_numpy(np.asarray(adata_query.obsm["X_pca_harmony"], dtype=np.float32)).float()}
    feature_dict_test = {section_key: input_dict}
    spatial_loc_dict_test = {section_key: np.asarray(adata_query.obsm["spatial"], dtype=np.float32)}

    final_embeddings_test = infer_embeddings(
        model,
        feature_dict_test,
        spatial_loc_dict_test,
        device,
        config["training"]["knn_neighbors_spatial"],
        config["training"]["knn_neighbors_feature"],
    )

    cosie_emb_meta = final_embeddings_test[section_key]

    del model, final_embeddings_test, input_dict, feature_dict_test, spatial_loc_dict_test
    gc.collect()

    # =========================================================
    # 5. Embedding PCA + label transfer
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 6. Run label transfer")
    print("=" * 80)

    C = np.load(str(centroid_path))
    pca_emb = joblib.load(str(emb_pca_path))

    embedding_query_pca_meta = pca_emb.transform(cosie_emb_meta)
    assigned_idx = pairwise_distances_argmin(embedding_query_pca_meta, C)
    assigned_labels_meta = np.arange(C.shape[0])[assigned_idx].astype(np.int32)

    del C, pca_emb, assigned_idx
    gc.collect()

    # =========================================================
    # 6. Save inferred AnnData
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 7. Save output AnnData")
    print("=" * 80)

    if use_metacell:
        cosie_emb_cell = broadcast_back(meta_id_per_cell, cosie_emb_meta).astype(np.float32, copy=False)
        X_ipca_cell = broadcast_back(meta_id_per_cell, embedding_query_pca_meta).astype(np.float32, copy=False)
        X_symphony_cell = broadcast_back(meta_id_per_cell, adata_query.obsm["X_pca_harmony"]).astype(np.float32, copy=False)
        X_input_pca_cell = broadcast_back(meta_id_per_cell, X_input_pca).astype(np.float32, copy=False)
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
        adata_out.uns["input_modality"] = modality
        adata_out.uns["use_metacell"] = bool(use_metacell)

        adata_meta_out = ad.AnnData(X=np.zeros((adata_query.n_obs, 0), dtype=np.float32))
        adata_meta_out.obsm["spatial"] = np.asarray(adata_query.obsm["spatial"], dtype=np.float32)
        adata_meta_out.obsm["X_cosie"] = cosie_emb_meta.astype(np.float32, copy=False)
        adata_meta_out.obsm["X_ipca"] = embedding_query_pca_meta.astype(np.float32, copy=False)
        adata_meta_out.obs["assigned_label"] = assigned_labels_meta.astype(np.int32, copy=False)
        adata_meta_out.uns["meta_id_per_cell"] = meta_id_per_cell.astype(np.int32, copy=False)
        adata_meta_out.uns["original_n_obs"] = int(spatial_cell.shape[0])
        adata_meta_out.uns["block_size"] = int(block_size)
        adata_meta_out.uns["input_modality"] = modality
        adata_meta_out.uns["use_metacell"] = True

        save_meta_h5ad_path = out_dir / "adata_query_metacell_inferred.h5ad"
        adata_meta_out.write(save_meta_h5ad_path)

    else:
        adata_out = ad.AnnData(X=np.zeros((adata_query.n_obs, 0), dtype=np.float32))
        adata_out.obsm["spatial"] = np.asarray(adata_query.obsm["spatial"], dtype=np.float32)
        adata_out.obsm["X_cosie"] = cosie_emb_meta.astype(np.float32, copy=False)
        adata_out.obsm["X_ipca"] = embedding_query_pca_meta.astype(np.float32, copy=False)
        adata_out.obsm["X_symphony"] = np.asarray(adata_query.obsm["X_pca_harmony"], dtype=np.float32)
        adata_out.obsm["X_input_pca"] = X_input_pca.astype(np.float32, copy=False)
        adata_out.obs["assigned_label"] = assigned_labels_meta.astype(np.int32, copy=False)
        adata_out.uns["input_modality"] = modality
        adata_out.uns["use_metacell"] = bool(use_metacell)

    # =========================================================
    # 7. Cluster -> cell-type
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 8. Obtain cell-type labels")
    print("=" * 80)

    cluster_key = "assigned_label"
    group_key = "celltype_labels"

    group_dict = {
        "Macrophages": [11],
        "Bronchus": [3],
        "Vessels": [8, 10],
        "Normal lung": [1, 4, 9, 13, 15, 18, 21, 23],
        "Pneumocytes": [17],
        "Tumor": [2, 5, 6, 7, 14, 16, 19, 20],
        "Fibrous tissue": [0, 12, 22],
        "Lymphoid aggregates": [24],
    }

    colormap = [
        [255, 127, 14],
        [188, 189, 34],
        [220, 20, 60],
        [173, 216, 230],
        [77, 175, 74],
        [148, 103, 189],
        [247, 182, 210],
        [139, 69, 19],
    ]

    legend_labels = [f"{name} (clusters {','.join(map(str, clusters))})" for name, clusters in group_dict.items()]

    assign_group_from_clusters(adata_out, cluster_key=cluster_key, group_dict=group_dict, new_key=group_key)

    save_h5ad_path = out_dir / "adata_query_inferred.h5ad"
    adata_out.write(save_h5ad_path)
    print(f"Saved final inferred AnnData to: {save_h5ad_path}")

    # =========================================================
    # 8. Plot
    # =========================================================
    print("\n" + "=" * 80)
    print("Step 9. Save final cell-type plot")
    print("=" * 80)

    fig_path = out_dir / "celltype_labels.png"

    visualize_superpixel_from_adata(
        adata_out,
        obs_key=group_key,
        colormap=colormap,
        legend_labels=legend_labels,
        figscale=200,
        save_path=fig_path,
    )

    print(f"Saved final plot to: {fig_path}")

    # =========================================================
    # 9. Cleanup
    # =========================================================
    del adata_query, adata_out, cosie_emb_meta, embedding_query_pca_meta, assigned_labels_meta, X_input_pca
    gc.collect()

    print("\n" + "=" * 80)
    print(f"All {modality} inference outputs saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()