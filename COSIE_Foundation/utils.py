import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import anndata as ad
import math
import scipy
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import random
import torch
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 
from annoy import AnnoyIndex
import hnswlib
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from scipy import sparse
from scipy.stats import entropy
from pandas.api.types import CategoricalDtype
import gc
from pathlib import Path
import pickle


def nn_approx(ds1, ds2, knn=10, metric='euclidean', n_trees=10, include_distances=False):

    """
    Efficiently find approximate K-nearest neighbors using the Annoy library.

    Parameters
    ----------
    ds1 : np.ndarray
        Query data of shape (n_query, dim), where neighbors will be searched for.
    
    ds2 : np.ndarray
        Reference data of shape (n_ref, dim), in which the neighbors are searched.
    
    knn : int, optional
        Number of neighbors to retrieve per query. Default is 10.
    
    metric : str, optional
        Distance metric used in Annoy. Must be one of: {'euclidean', 'manhattan', 'angular', 'hamming', 'dot'}. Default is 'euclidean'.
    
    n_trees : int, optional
        Number of trees used to build the Annoy index. Higher values increase accuracy at the cost of indexing time. Default is 10.
    
    include_distances : bool, optional
        Whether to also return distances to the nearest neighbors. Default is False.

    Returns
    -------
    ind : np.ndarray
        If `include_distances` is False, returns an array of shape (n_query, knn) with indices of nearest neighbors.

    tuple of (ind, dist) : (np.ndarray, np.ndarray)
        If `include_distances` is True, returns a tuple:
        
        - `ind` : array of shape (n_query, knn) with indices of nearest neighbors.
        - `dist` : array of shape (n_query, knn) with corresponding distances.
    """


    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind, dist = [], []
    for i in range(ds1.shape[0]):
        i_ind, i_dist = a.get_nns_by_vector(ds1[i, :], knn, search_k=-1, include_distances=True)
        ind.append(i_ind)
        dist.append(i_dist)
    ind = np.array(ind)
    
    if include_distances:
        return ind, np.array(dist)
    else:
        # return ind.flatten()
        return ind


def setup_seed(seed=8, mode='fast'):

    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters
    ----------
    seed : int, optional
        The random seed to be set for all relevant libraries. Default is 8.
    mode : str, optional
        Controls how strictly reproducibility is enforced. Must be one of {'fast', 'strict'}.
        
        - 'fast': Ensures reproducibility in most cases, with minimal performance impact.  
        - 'strict': Enforces full determinism across all operations (including CUDA),
          but may significantly slow down certain models.

        It is recommended to use 'strict' only when exact reproducibility across runs is required.

    Returns
    -------
    None
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # control cuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # strict mode
    if mode == 'strict':
        ### This will slow down the learning process
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'





def compute_knn_graph(input_data, n_neighbors):

    """
    Construct a k-nearest neighbors (k-NN) graph based on an input feature matrix.

    Parameters
    ----------
    input_data : np.ndarray
        An array of shape (n_cells, n_features) representing input features of cells,
        which can be spatial coordinates or feature vectors.
    
    n_neighbors : int
        Number of nearest neighbors to connect for each node.

    Returns
    -------
    edge_index : torch.LongTensor
        A tensor of shape (2, num_edges) representing the edges of the graph. Each column (i, j) represents an edge from node i to its j-th nearest neighbor.
    """

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(input_data)  
    _ , indices = nbrs.kneighbors(input_data)
    x = indices[:, 0].repeat(n_neighbors+1)
    y = indices[:, 0:].flatten() 
    edge_index = np.vstack((x, y))    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index






def construct_knn_graph_hnsw(data, k=20, space='l2'):

    """
    Efficiently compute approximate k-nearest neighbor (k-NN) graph using the hnswlib library.
    This method is suitable for large-scale datasets.

    Parameters
    ----------
    data : torch.Tensor
        A tensor of shape (n_cells, n_features) representing the input feature matrix.
    
    k : int, optional
        Number of nearest neighbors to retrieve for each sample. Default is 20.
    
    space : str, optional
        Distance metric used to build the index. Must be one of {'l2', 'ip', 'cosine'}.
        
        - 'l2': Euclidean distance  
        - 'ip': Inner product  
        - 'cosine': Cosine similarity  

        Default is 'l2'.

    Returns
    -------
    edge_index : torch.LongTensor
        A tensor of shape (2, n_edges) representing the edges of the graph.
        Each column represents an edge from source node to target node.
    """

    random.seed(42)
    np.random.seed(42)
    data = data.cpu().numpy().astype(np.float32)
    num_samples, dim = data.shape

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=200, M=16)
    p.add_items(data)

    p.set_ef(50)

    indices, distances = p.knn_query(data, k=k)
    
    row_indices = np.repeat(np.arange(num_samples), k)
    col_indices = indices.flatten()
    edge_index = torch.tensor(np.vstack((row_indices, col_indices)), dtype=torch.long)
    
    return edge_index



def compute_neighborhood_embedding(edge_index_spatial: torch.Tensor, embedding_matrix: torch.Tensor, device) -> torch.Tensor:

    """
    Compute the neighborhood embedding of each cell based on its spatial k-nearest neighbor graph.

    Parameters
    ----------
    edge_index_spatial : torch.Tensor
        A tensor of shape (2, num_edges), where each column represents a directed edge from source node to target node in the spatial graph.
    
    embedding_matrix : torch.Tensor
        A tensor of shape (n_cells, dim) representing the embedding of each cell.
    
    device : torch.device
        The computation device, e.g., `torch.device('cuda')` or `torch.device('cpu')`.

    Returns
    -------
    neighborhood_embedding_matrix : torch.Tensor
        A tensor of shape (n_cells, dim) representing the average of each cell’s neighbors' embeddings.
    """

    
    # Get the number of nodes and embedding dimension
    num_nodes = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    
    # Initialize the neighborhood embedding matrix and count tensor
    neighborhood_embedding_matrix = torch.zeros(num_nodes, embedding_dim, device=device)
    neighbor_count = torch.zeros(num_nodes, dtype=torch.float32, device=device)

    # Use edge_index_spatial to index into the embedding matrix
    source_nodes = edge_index_spatial[0]  # Get source nodes
    target_nodes = edge_index_spatial[1]  # Get target nodes

    # Accumulate the embeddings from target nodes to the corresponding source nodes
    neighborhood_embedding_matrix.index_add_(0, source_nodes, embedding_matrix[target_nodes])
    
    # Count neighbors for each source node
    neighbor_count.index_add_(0, source_nodes, torch.ones_like(target_nodes, dtype=torch.float32, device=device))

    # Avoid division by zero by replacing zero counts with one
    neighbor_count[neighbor_count == 0] = 1

    # Compute the average by dividing by the count of neighbors
    neighborhood_embedding_matrix /= neighbor_count.view(-1, 1)
    

    return neighborhood_embedding_matrix

    



# for large-scale integration

def subsample_by_kmeans(
    X,
    cluster_num: int,
    sample_ratio_each_cluster=0.1,
    random_state: int = 0,
):
    """
    Perform KMeans clustering on the given matrix X and subsample the original AnnData:
    for each cluster, select the closest cells to the cluster center.

    Parameters
    ----------

    X : array-like (n_cells, n_features)
        The matrix used for clustering (e.g., PCA result).
    cluster_num : int
        Number of clusters (K).
    sample_ratio_each_cluster : float, dict[int,float], or int
        - float: sampling ratio (0-1) applied to each cluster
        - dict: specify sampling ratio for each cluster individually, e.g. {0:0.05, 1:0.2, ...}
        - int: fixed number of samples per cluster (at least 1 and not more than cluster size)
    random_state : int
        Random seed (for KMeans initialization).

    Returns
    -------

    selected_indices : np.ndarray
        The row indices of selected cells in the original AnnData (sorted).
    labels : np.ndarray
        Cluster labels for all cells (length = n_cells).
    centers : np.ndarray
        KMeans cluster centers with shape (K, n_features).
    """
    # Ensure X is a dense numpy array (required by sklearn KMeans)
    if sparse.issparse(X):
        X_np = X.toarray()
    elif hasattr(X, "detach"):  # torch.Tensor
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    X_np = check_array(X_np, accept_sparse=False, ensure_min_samples=2)
    n = X_np.shape[0]
    # if n != adata.n_obs:
    #     raise ValueError(f"X has {n} rows but adata has {adata.n_obs} cells.")

    # Run standard KMeans
    kmeans = KMeans(n_clusters=cluster_num, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_np)
    centers = kmeans.cluster_centers_

    selected_indices = []

    for c in range(cluster_num):
        cluster_idx = np.where(labels == c)[0]
        if cluster_idx.size == 0:
            continue  # skip empty cluster (rare)

        # Determine sample size for this cluster
        if isinstance(sample_ratio_each_cluster, dict):
            ratio = float(sample_ratio_each_cluster.get(c, 0.0))
            sample_size = int(np.floor(cluster_idx.size * max(0.0, min(1.0, ratio))))
        elif isinstance(sample_ratio_each_cluster, (int, np.integer)):
            sample_size = int(sample_ratio_each_cluster)
        else:  # float
            ratio = float(sample_ratio_each_cluster)
            sample_size = int(np.floor(cluster_idx.size * max(0.0, min(1.0, ratio))))

        # Ensure boundaries: at least 1 and not more than cluster size;
        # if set to 0, skip this cluster
        if isinstance(sample_ratio_each_cluster, (int, np.integer)) and sample_size <= 0:
            sample_size = 1
        if sample_size == 0:
            continue
        sample_size = min(sample_size, cluster_idx.size)

        # Compute Euclidean distance to cluster center
        # Use argpartition to get the nearest k points, then sort those
        diffs = X_np[cluster_idx] - centers[c]
        dists = np.linalg.norm(diffs, axis=1)

        kth = np.argpartition(dists, sample_size - 1)[:sample_size]
        nearest_sorted = kth[np.argsort(dists[kth])]

        selected_indices.extend(cluster_idx[nearest_sorted])

    # selected_indices = np.unique(np.asarray(selected_indices, dtype=int))
    # # Build the subsampled AnnData and save the original indices
    # new_adata_sub = adata[selected_indices].copy()
    # new_adata_sub.obs["original_index"] = selected_indices.tolist()

    return selected_indices, labels, centers



def subset_cosie_inputs(feature_dict, spatial_loc_dict, data_dict_processed, sub_indices_dict):
    """
    Subset COSIE inputs using pre-computed sub_indices for each section.

    Parameters
    ----------
    feature_dict : dict
        Original feature dictionary {section: {modality: tensor}}.
    spatial_loc_dict : dict
        Original spatial location dictionary {section: np.ndarray (n_cells, 2)}.
    data_dict_processed : dict
        Original processed AnnData dictionary {modality: [adata_s1, adata_s2, ...]}.
    sub_indices_dict : dict
        Dictionary {section: np.ndarray of selected indices}.
        e.g., {"s1": sub_indices1, "s2": sub_indices2}

    Returns
    -------
    feature_dict_sub : dict
        Subset feature dictionary.
    spatial_loc_dict_sub : dict
        Subset spatial dictionary.
    data_dict_processed_sub : dict
        Subset AnnData dictionary.
    """
    # 1. Subset feature_dict
    feature_dict_sub = {}
    for section, modalities in feature_dict.items():
        if section not in sub_indices_dict:
            continue
        idx = sub_indices_dict[section]
        feature_dict_sub[section] = {
            mod: feat[idx] for mod, feat in modalities.items()
        }

    # 2. Subset spatial_loc_dict
    spatial_loc_dict_sub = {
        section: spatial[idx]
        for section, spatial in spatial_loc_dict.items()
        if section in sub_indices_dict
        for idx in [sub_indices_dict[section]]
    }

    # 3. Subset data_dict_processed
    data_dict_processed_sub = {}
    for modality, adata_list in data_dict_processed.items():
        new_list = []
        for sec_idx, adata in enumerate(adata_list):
            section_name = f"s{sec_idx+1}"
            if adata is None or section_name not in sub_indices_dict:
                new_list.append(None)
            else:
                idx = sub_indices_dict[section_name]
                new_list.append(adata[idx].copy())
        data_dict_processed_sub[modality] = new_list

    return feature_dict_sub, spatial_loc_dict_sub, data_dict_processed_sub



def cluster_entropy(labels):
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)





def find_max_entropy(entropies, start, end):
    """
    entropies: 56-length list
    start/end: 1-based index (e.g., 1,18)
    """
    sub = entropies[start-1 : end]    
    max_val = max(sub)
    max_idx = sub.index(max_val) + start   
    
    return max_idx, max_val




def reconstruct_metacell_to_original_new(meta_map, adata_metacell, metacell_embedding):

    original_cell_num = adata_metacell.uns['original_cell_num']
    original_embedding = np.zeros((original_cell_num, metacell_embedding.shape[1]))
    for meta_idx, ori_indices in enumerate(meta_map):
        original_embedding[ori_indices] = metacell_embedding[meta_idx]

    return original_embedding






def infer_embeddings(model, feature_dict, spatial_loc_dict, device, k_neighs_spatial, k_neighs_feature):
    model.eval()
    model.to(device)

    final_embeddings = {}

    with torch.no_grad():
        for section, modalities in feature_dict.items():
            print(f"Processing section: {section}")

            # construct spatial graph 
            spatial_knn = compute_knn_graph(spatial_loc_dict[section], k_neighs_spatial).to(device)

            embeddings = {}
            for modality, features in modalities.items():
                feature_knn = construct_knn_graph_hnsw(features, k_neighs_feature).to(device)
                features = features.to(device)

                combined_knn = torch.cat([spatial_knn, feature_knn], dim=1)

                encoder = model.autoencoders[modality]
                z = encoder.encoder(features, combined_knn)
                embeddings[modality] = z

            # unify modal embedding
            recovered_embeddings = {}
            for mod in model.all_modalities:
                if mod in embeddings:
                    recovered_embeddings[mod] = embeddings[mod]
                else:
                    # use predictor to recover missing modality
                    candidate_embeddings = []
                    for src_mod in embeddings.keys():
                        predictor_key = f"{src_mod}_to_{mod}"
                        if predictor_key in model.predictors:
                            candidate_embeddings.append(
                                model.predictors[predictor_key](embeddings[src_mod])
                            )
                    if len(candidate_embeddings) > 0:
                        recovered_embeddings[mod] = torch.mean(torch.stack(candidate_embeddings), dim=0)
                    else:
                        recovered_embeddings[mod] = torch.zeros_like(next(iter(embeddings.values())))

            concatenated_embedding = torch.cat(
                [recovered_embeddings[mod] for mod in model.all_modalities], dim=1
            )
            neighborhood_embedding = compute_neighborhood_embedding(
                spatial_knn, concatenated_embedding, device
            )
            bi_embedding = (concatenated_embedding + neighborhood_embedding) * 0.5

            final_embeddings[section] = bi_embedding.cpu().numpy()
            del embeddings, recovered_embeddings, concatenated_embedding, neighborhood_embedding, bi_embedding
            torch.cuda.empty_cache() 

    return final_embeddings




# for cosie-foundation inference

def build_metacells_grid_fast(adata, block_size=2, spatial_key="spatial"):
    """
    Build metacells by grouping block_size x block_size adjacent grid cells.
    Only uses spatial coords; does not touch adata.X.

    Stores:
      - adata_meta.uns["meta_id_per_cell"] : int32 array of length n_original
      - adata_meta.uns["original_cell_num"]
      - adata_meta.uns["block_size"]
    """
    spatial = np.asarray(adata.obsm[spatial_key])
    y = spatial[:, 0]
    x = spatial[:, 1]

    # Estimate grid step (assumes near-regular grid)
    uniq_y = np.unique(y)
    uniq_x = np.unique(x)
    dy = np.diff(uniq_y)
    dx = np.diff(uniq_x)
    step_y = np.min(dy[dy > 0]) if np.any(dy > 0) else 1.0
    step_x = np.min(dx[dx > 0]) if np.any(dx > 0) else 1.0

    y0, x0 = y.min(), x.min()
    grid_y = np.rint((y - y0) / step_y).astype(np.int64)
    grid_x = np.rint((x - x0) / step_x).astype(np.int64)

    block_y = grid_y // block_size
    block_x = grid_x // block_size

    # Pair -> single key for grouping
    key = (block_y << 32) ^ (block_x & 0xFFFFFFFF)

    order = np.argsort(key)
    key_sorted = key[order]

    uniq_key, start = np.unique(key_sorted, return_index=True)
    n_meta = uniq_key.size
    counts = np.diff(np.r_[start, key_sorted.size]).astype(np.int64)

    meta_id_per_cell = np.empty(adata.n_obs, dtype=np.int32)
    meta_id_per_cell[order] = np.repeat(np.arange(n_meta, dtype=np.int32), counts)

    # Metacell spatial mean via bincount
    denom = np.bincount(meta_id_per_cell, minlength=n_meta).astype(np.float32)
    meta_y = np.bincount(meta_id_per_cell, weights=y, minlength=n_meta) / denom
    meta_x = np.bincount(meta_id_per_cell, weights=x, minlength=n_meta) / denom
    meta_spatial = np.vstack([meta_y, meta_x]).T.astype(np.float32)

    adata_meta = ad.AnnData(X=np.zeros((n_meta, 0), dtype=np.float32))
    adata_meta.obsm[spatial_key] = meta_spatial
    adata_meta.uns["meta_id_per_cell"] = meta_id_per_cell
    adata_meta.uns["original_cell_num"] = int(adata.n_obs)
    adata_meta.uns["block_size"] = int(block_size)
    return adata_meta


def aggregate_X_to_metacell_mean_dense(X, meta_id_per_cell, n_meta, bs=20000):
    """
    Stream through dense X (possibly backed) and compute metacell mean features.

    X: supports row slicing X[s:e] returning array-like (dense).
    meta_id_per_cell: int32 array len n_cells
    n_meta: number of metacells
    """
    n = meta_id_per_cell.shape[0]
    d = X.shape[1]

    meta_sum = np.zeros((n_meta, d), dtype=np.float32)
    meta_cnt = np.bincount(meta_id_per_cell, minlength=n_meta).astype(np.float32)

    for s in range(0, n, bs):
        e = min(s + bs, n)
        Xb = np.asarray(X[s:e], dtype=np.float32)  # small batch
        gid = meta_id_per_cell[s:e].astype(np.int64)
        np.add.at(meta_sum, gid, Xb)
        del Xb, gid
        if (s // bs) % 20 == 0:
            gc.collect()

    meta_mean = meta_sum / meta_cnt[:, None]
    return meta_mean


def broadcast_back(meta_id_per_cell, meta_array):
    """Vectorized broadcast: (n_meta, d) -> (n_cells, d)"""
    return meta_array[meta_id_per_cell]





def visualize_superpixel_from_adata(
    adata,
    obs_key,
    vis_basis="spatial",
    colormap=None,
    legend_labels=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_path=None,
    dpi=300,
    remove_title=False,
    remove_legend=False,
    remove_spine=False,
    figscale=35,
):
    import numpy as np

    # ---- coords ----
    coords = adata.obsm[vis_basis].copy()
    if swap_xy:
        coords = coords[:, [1, 0]]
    coords = coords.astype(int)

    if offset:
        coords -= coords.min(axis=0)

    # ---- labels ----
    labels = adata.obs[obs_key]

    cluster_names = None

    if hasattr(labels.dtype, "categories"):
        # categorical labels
        label_codes = labels.cat.codes.values

        if legend_labels is not None:
            cluster_names = legend_labels
            num_clusters = len(legend_labels)
        else:
            cluster_names = list(labels.cat.categories)
            num_clusters = len(cluster_names)

    else:
        # numeric labels
        label_codes = labels.astype(int).values
        num_clusters = int(label_codes.max()) + 1
        cluster_names = (
            legend_labels
            if legend_labels is not None
            else [f"Cluster {i}" for i in range(num_clusters)]
        )

    # ---- build image ----
    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), fill_value=-1, dtype=int)

    for (y, x), lab in zip(coords, label_codes):
        if lab >= 0:
            image[y, x] = lab

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    # ---- plot ----
    plot_histology_clusters(
        he_clusters_image=image,
        num_he_clusters=num_clusters,
        section_title=obs_key,
        colormap=colormap,
        cluster_names=cluster_names,
        save_path=save_path,
        dpi=dpi,
        figscale=figscale,
        remove_title=remove_title,
        remove_legend=remove_legend,
        remove_spine=remove_spine,
    )


def plot_histology_clusters(
    he_clusters_image,
    num_he_clusters,
    section_title=None,
    colormap=None,
    cluster_names=None,
    save_path=None,
    figscale=35,
    remove_title=False,
    remove_legend=False,
    remove_spine=False,
    dpi=300,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    from matplotlib.colors import to_rgb

    # ---- colors ----
    if colormap is None:
        color_list = [
            [255,127,14],[44,160,44],[214,39,40],[148,103,189],
            [140,86,75],[227,119,194],[127,127,127],[188,189,34],
            [23,190,207],[174,199,232],[255,187,120],[152,223,138],
            [255,152,150],[197,176,213],[196,156,148],[247,182,210],
            [199,199,199],[219,219,141],[158,218,229],[16,60,90],
            [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
            [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0],
        ]
    elif isinstance(colormap, list):
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap)
        color_list = [
            [int(255 * c) for c in to_rgb(cmap(i))]
            for i in range(num_he_clusters)
        ]

    if len(color_list) < num_he_clusters:
        raise ValueError(
            f"Color list has {len(color_list)} colors but "
            f"{num_he_clusters} clusters are present."
        )

    # ---- RGB image ----
    h, w = he_clusters_image.shape
    image_rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

    for c in range(num_he_clusters):
        image_rgb[he_clusters_image == c] = color_list[c]

    # ---- plot ----
    plt.figure(figsize=(w // figscale, h // figscale))
    if not remove_title:
        plt.title(section_title or "Histology Clusters", fontsize=18)

    plt.imshow(image_rgb, interpolation="none")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ---- legend ----
    if not remove_legend:
        legend_elements = []
        for i in range(num_he_clusters):
            label = (
                cluster_names[i]
                if cluster_names is not None and i < len(cluster_names)
                else f"Cluster {i}"
            )
            legend_elements.append(
                patches.Patch(
                    facecolor=np.array(color_list[i]) / 255,
                    label=str(label),
                )
            )

        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=12,
        )

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def assign_group_from_clusters(adata, cluster_key, group_dict, new_key="group_label"):
    """Map cluster ids -> group names, save to adata.obs[new_key]."""
    cluster_ids = adata.obs[cluster_key].astype(int).values
    labels = np.array([None] * adata.n_obs, dtype=object)

    for group_name, clusters in group_dict.items():
        mask = np.isin(cluster_ids, clusters)
        labels[mask] = group_name

    if np.any(labels == None):
        missing_clusters = np.unique(cluster_ids[labels == None])
        raise ValueError(f"Some clusters are not assigned to any group: {missing_clusters}")

    categories = list(group_dict.keys())
    adata.obs[new_key] = labels
    adata.obs[new_key] = adata.obs[new_key].astype(CategoricalDtype(categories=categories))
    
    return adata.obs[new_key]




# For virtual prediction

def load_ref_bundle_shared(bundle_dir, metric="euclidean"):
    bundle_dir = Path(bundle_dir)

    with open(bundle_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open(bundle_dir / "secs_sorted.pkl", "rb") as f:
        secs_sorted = pickle.load(f)

    with open(bundle_dir / "X_shapes.pkl", "rb") as f:
        shapes = pickle.load(f)

    with open(bundle_dir / "emb_dim.pkl", "rb") as f:
        d = pickle.load(f)

    pool_sec = np.load(bundle_dir / "pool_sec.npy").astype(np.int32, copy=False)
    pool_row = np.load(bundle_dir / "pool_row.npy").astype(np.int32, copy=False)

    annoy = AnnoyIndex(int(d), metric)
    annoy.load(str(bundle_dir / "annoy_index.ann"))

    xdir = bundle_dir / "X_by_section"
    X_src_list = []
    for sec in secs_sorted:
        shp = shapes[sec]
        mm_path = xdir / f"{sec}.float32.dat"
        X_sec = np.memmap(mm_path, dtype="float32", mode="r", shape=shp)
        X_src_list.append(X_sec)

    return X_src_list, annoy, pool_sec, pool_row, feature_names, d, secs_sorted



def impute_shared_features_annoy(
    X_src_list,
    annoy_index,
    pool_sec,
    pool_row,
    query_emb,
    feature_names,
    K=50,
    out_prefix=None,
    block_query=8192,
    verbose=1
):
    query_emb = np.asarray(query_emb, dtype=np.float32, order="C")
    n_q = query_emb.shape[0]
    n_f = len(feature_names)

    if out_prefix is not None:
        out_dir = os.path.dirname(out_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mm_path = out_prefix + f".K{K}.float32.dat"
        out = np.memmap(mm_path, dtype="float32", mode="w+", shape=(n_q, n_f))
    else:
        out = np.zeros((n_q, n_f), dtype=np.float32)

    for s in range(0, n_q, block_query):
        e = min(s + block_query, n_q)
        if verbose:
            print(f"[Impute] query block {s}:{e} / {n_q}")

        for i in range(s, e):
            nn_ids = annoy_index.get_nns_by_vector(query_emb[i], K, include_distances=False)
            nn_ids = np.asarray(nn_ids, dtype=np.int32)

            sids = pool_sec[nn_ids]
            rows = pool_row[nn_ids]

            vals = []
            for sid in np.unique(sids):
                mask = (sids == sid)
                r = rows[mask]
                vals.append(X_src_list[int(sid)][r, :])

            out[i, :] = np.concatenate(vals, axis=0).mean(axis=0)

    if isinstance(out, np.memmap):
        out.flush()
        return out.filename, feature_names
    return out, feature_names


##