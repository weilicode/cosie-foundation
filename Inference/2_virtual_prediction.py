#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from annoy import AnnoyIndex
from COSIE_Foundation.utils import *


def parse_args():
    parser = argparse.ArgumentParser("COSIE virtual prediction pipeline")

    parser.add_argument(
        "--inference-root",
        type=str,
        required=True,
        help="Inference output directory containing adata_query_inferred.h5ad"
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        required=True,
        help="Reference bundle directory for virtual prediction"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=200,
        help="Number of nearest neighbors for imputation. Default: 200"
    )
    parser.add_argument(
        "--block-query",
        type=int,
        default=1024,
        help="Query block size for imputation. Default: 1024"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Paths / settings from args
    # --------------------------------------------------------
    inference_root = Path(args.inference_root)
    bundle_dir = Path(args.bundle_dir)

    cell_h5ad = inference_root / "adata_query_inferred.h5ad"
    meta_h5ad = inference_root / "adata_query_metacell_inferred.h5ad"   # optional
    out_h5ad = inference_root / "adata_query_predicted.h5ad"

    K = args.K
    block_query = args.block_query
    verbose = 1
    cleanup_mm = True

    if not inference_root.exists():
        raise FileNotFoundError(f"inference_root not found: {inference_root}")

    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle_dir not found: {bundle_dir}")

    if not cell_h5ad.exists():
        raise FileNotFoundError(f"Missing query h5ad: {cell_h5ad}")

    # --------------------------------------------------------
    # Load reference bundle
    # --------------------------------------------------------
    X_src_list, annoy, pool_sec, pool_row, feature_names, d, secs_sorted = load_ref_bundle_shared(bundle_dir)
    n_f = len(feature_names)

    print("\n========== Query prediction ==========")
    print(f"[Input] cell_h5ad: {cell_h5ad}")
    print(f"[Input] meta_h5ad: {meta_h5ad} | exists={meta_h5ad.exists()}")
    print(f"[Ref] embedding dim = {d}")
    print(f"[Ref] total predicted features = {n_f}")

    use_metacell = meta_h5ad.exists()
    print(f"[Mode] use_metacell = {use_metacell}")

    # ========================================================
    # A. metacell mode
    # ========================================================
    if use_metacell:
        adata_meta = sc.read_h5ad(str(meta_h5ad))

        if "X_ipca" not in adata_meta.obsm:
            raise KeyError(f"Missing obsm['X_ipca'] in metacell h5ad: {meta_h5ad}")

        query_emb_meta = np.asarray(adata_meta.obsm["X_ipca"], dtype=np.float32, order="C")

        if query_emb_meta.shape[1] != d:
            raise ValueError(f"metacell X_ipca dim {query_emb_meta.shape[1]} != ref dim {d}")

        n_meta = query_emb_meta.shape[0]
        print(f"[Metacell] query_emb_meta shape = {query_emb_meta.shape}")

        # out_prefix_meta = str(inference_root / "imputed_rna_adt_metacell")
        # mm_path_meta = out_prefix_meta + f".K{K}.float32.dat"

        # if os.path.exists(mm_path_meta):
        #     print(f"[Metacell] memmap exists, reuse: {mm_path_meta}")
        # else:
        #     mm_path_meta, _ = impute_shared_features_annoy(
        #         X_src_list=X_src_list,
        #         annoy_index=annoy,
        #         pool_sec=pool_sec,
        #         pool_row=pool_row,
        #         query_emb=query_emb_meta,
        #         feature_names=feature_names,
        #         K=K,
        #         out_prefix=out_prefix_meta,
        #         block_query=block_query,
        #         verbose=verbose
        #     )
        #     print(f"[Metacell] saved memmap: {mm_path_meta}")

        out_prefix_meta = str(inference_root / "imputed_rna_adt_metacell")
        mm_path_meta = out_prefix_meta + f".K{K}.float32.dat"
        done_path_meta = out_prefix_meta + f".K{K}.done"
        
        if os.path.exists(mm_path_meta) and os.path.exists(done_path_meta):
            print(f"[Metacell] completed memmap exists, reuse: {mm_path_meta}")
        else:
            # stale/incomplete file cleanup
            if os.path.exists(mm_path_meta):
                print(f"[Metacell] found incomplete memmap, deleting: {mm_path_meta}")
                os.remove(mm_path_meta)
            if os.path.exists(done_path_meta):
                os.remove(done_path_meta)
        
            mm_path_meta, _ = impute_shared_features_annoy(
                X_src_list=X_src_list,
                annoy_index=annoy,
                pool_sec=pool_sec,
                pool_row=pool_row,
                query_emb=query_emb_meta,
                feature_names=feature_names,
                K=K,
                out_prefix=out_prefix_meta,
                block_query=block_query,
                verbose=verbose
            )
        
            # write completion marker only after successful finish
            with open(done_path_meta, "w") as f:
                f.write("done\n")
        
            print(f"[Metacell] saved memmap: {mm_path_meta}")
            print(f"[Metacell] saved done flag: {done_path_meta}")

        # --------------------------------------------
        # Broadcast back to cell-level
        # --------------------------------------------
        adata_cell = sc.read_h5ad(str(cell_h5ad))

        if "X_ipca" not in adata_cell.obsm:
            raise KeyError(f"Missing obsm['X_ipca'] in cell h5ad: {cell_h5ad}")

        n_cell = adata_cell.n_obs


        # print("[Broadcast] inferring cell-to-metacell mapping by nearest metacell in X_ipca space ...")

        # from sklearn.metrics import pairwise_distances_argmin
        # meta_id_per_cell = pairwise_distances_argmin(
        #     np.asarray(adata_cell.obsm["X_ipca"], dtype=np.float32),
        #     np.asarray(adata_meta.obsm["X_ipca"], dtype=np.float32),
        # ).astype(np.int64)
        if "meta_id_per_cell" not in adata_meta.uns:
            raise KeyError("Missing uns['meta_id_per_cell'] in adata_query_metacell_inferred.h5ad")
        
        meta_id_per_cell = np.asarray(adata_meta.uns["meta_id_per_cell"], dtype=np.int64)
        if meta_id_per_cell.shape[0] != n_cell:
            raise ValueError(
                f"meta_id_per_cell length {meta_id_per_cell.shape[0]} != n_cell {n_cell}"
            )
        print(f"[Broadcast] loaded meta_id_per_cell from metacell h5ad, shape={meta_id_per_cell.shape}")

        X_meta_mm = np.memmap(mm_path_meta, dtype="float32", mode="r", shape=(n_meta, n_f))

        print(f"[Broadcast] allocating cell matrix: ({n_cell}, {n_f})")
        X_cell = X_meta_mm[meta_id_per_cell, :]

        adata_out = ad.AnnData(
            X=np.asarray(X_cell, dtype=np.float32, order="C"),
            obs=adata_cell.obs.copy(),
            var=pd.DataFrame(index=pd.Index(feature_names, dtype=str))
        )

        if "spatial" in adata_cell.obsm:
            adata_out.obsm["spatial"] = adata_cell.obsm["spatial"].copy()

        # optionally keep useful low-d info
        if "X_ipca" in adata_cell.obsm:
            adata_out.obsm["X_ipca"] = adata_cell.obsm["X_ipca"].copy()

        adata_out.write_h5ad(str(out_h5ad))
        print(f"[Done] saved: {out_h5ad}")

        del adata_out, X_cell, X_meta_mm, adata_cell, adata_meta, query_emb_meta, meta_id_per_cell
        gc.collect()

        if cleanup_mm:
            try:
                os.remove(mm_path_meta)
                print(f"[Cleanup] deleted memmap: {mm_path_meta}")
            except Exception as e:
                print(f"[Cleanup-WARN] failed to delete memmap: {mm_path_meta} | {repr(e)}")

    # ========================================================
    # B. cell-level mode
    # ========================================================
    else:
        adata_cell = sc.read_h5ad(str(cell_h5ad))

        if "X_ipca" not in adata_cell.obsm:
            raise KeyError(f"Missing obsm['X_ipca'] in cell h5ad: {cell_h5ad}")

        query_emb = np.asarray(adata_cell.obsm["X_ipca"], dtype=np.float32, order="C")

        if query_emb.shape[1] != d:
            raise ValueError(f"cell X_ipca dim {query_emb.shape[1]} != ref dim {d}")

        # out_prefix = str(inference_root / "imputed_rna_adt")
        # mm_path, _ = impute_shared_features_annoy(
        #     X_src_list=X_src_list,
        #     annoy_index=annoy,
        #     pool_sec=pool_sec,
        #     pool_row=pool_row,
        #     query_emb=query_emb,
        #     feature_names=feature_names,
        #     K=K,
        #     out_prefix=out_prefix,
        #     block_query=block_query,
        #     verbose=verbose
        # )
        out_prefix = str(inference_root / "imputed_rna_adt")
        mm_path = out_prefix + f".K{K}.float32.dat"
        done_path = out_prefix + f".K{K}.done"
        
        if os.path.exists(mm_path) and os.path.exists(done_path):
            print(f"[Cell] completed memmap exists, reuse: {mm_path}")
        else:
            if os.path.exists(mm_path):
                print(f"[Cell] found incomplete memmap, deleting: {mm_path}")
                os.remove(mm_path)
            if os.path.exists(done_path):
                os.remove(done_path)
        
            mm_path, _ = impute_shared_features_annoy(
                X_src_list=X_src_list,
                annoy_index=annoy,
                pool_sec=pool_sec,
                pool_row=pool_row,
                query_emb=query_emb,
                feature_names=feature_names,
                K=K,
                out_prefix=out_prefix,
                block_query=block_query,
                verbose=verbose
            )
        
            with open(done_path, "w") as f:
                f.write("done\n")
        
            print(f"[Cell] saved memmap: {mm_path}")
            print(f"[Cell] saved done flag: {done_path}")


        n_cell = query_emb.shape[0]
        X_imp_mm = np.memmap(mm_path, dtype="float32", mode="r", shape=(n_cell, n_f))
        X_imp = np.asarray(X_imp_mm)

        adata_out = ad.AnnData(
            X=np.asarray(X_imp, dtype=np.float32, order="C"),
            obs=adata_cell.obs.copy(),
            var=pd.DataFrame(index=pd.Index(feature_names, dtype=str))
        )

        if "spatial" in adata_cell.obsm:
            adata_out.obsm["spatial"] = adata_cell.obsm["spatial"].copy()

        if "X_ipca" in adata_cell.obsm:
            adata_out.obsm["X_ipca"] = adata_cell.obsm["X_ipca"].copy()

        adata_out.write_h5ad(str(out_h5ad))
        print(f"[Done] saved: {out_h5ad}")

        del adata_out, X_imp, X_imp_mm, adata_cell, query_emb
        gc.collect()

        if cleanup_mm:
            try:
                os.remove(mm_path)
                print(f"[Cleanup] deleted memmap: {mm_path}")
            except Exception as e:
                print(f"[Cleanup-WARN] failed to delete memmap: {mm_path} | {repr(e)}")

    print("\nAll prediction outputs finished.")



if __name__ == "__main__":
    main()



##