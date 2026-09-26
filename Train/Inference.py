import os
import pickle
import argparse
import torch

from COSIE_Foundation.data_preprocessing import *
from COSIE_Foundation.utils import *
from COSIE_Foundation.configure import get_default_config
from COSIE_Foundation.COSIE_framework import COSIE_model
from COSIE_Foundation.downstream_analysis import *


def parse_args():
    parser = argparse.ArgumentParser("COSIE inference pipeline")
    parser.add_argument("--project-root", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================================================
    # 0. Basic settings
    # =========================================================
    config = get_default_config()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # =========================================================
    # 1. Paths
    # =========================================================
    project_root = args.project_root

    data_root = os.path.join(project_root, "Data_preprocessing")
    training_root = os.path.join(project_root, "Training")
    embedding_root = os.path.join(project_root, "Embedding")

    os.makedirs(embedding_root, exist_ok=True)

    checkpoint_path = os.path.join(
        training_root,
        "cosie_trained.pt"
    )

    cell_embedding_pkl = os.path.join(
        embedding_root,
        "final_embeddings_cell.pkl"
    )

    # =========================================================
    # 2. Load full data
    # =========================================================
    print("\nLoading full data...")

    with open(
        os.path.join(data_root, "feature_dict_concat.pkl"),
        "rb"
    ) as f:
        feature_dict = pickle.load(f)

    with open(
        os.path.join(data_root, "spatial_loc_dict.pkl"),
        "rb"
    ) as f:
        spatial_loc_dict = pickle.load(f)

    print("Loaded feature_dict sections:", len(feature_dict))
    print("Loaded spatial_loc_dict sections:", len(spatial_loc_dict))

    # =========================================================
    # 3. Load trained model
    # =========================================================
    print("\nLoading trained model...")

    model = COSIE_model(config, feature_dict)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")

    # =========================================================
    # 4. Full-data inference
    # =========================================================
    print("\nRunning full-data inference...")

    with torch.no_grad():
        final_embeddings = infer_embeddings(
            model,
            feature_dict,
            spatial_loc_dict,
            device,
            config["training"]["knn_neighbors_spatial"],
            config["training"]["knn_neighbors_feature"]
        )

    # =========================================================
    # 5. Save
    # =========================================================
    with open(cell_embedding_pkl, "wb") as f:
        pickle.dump(final_embeddings, f)

    print(f"Saved full-data embeddings to: {cell_embedding_pkl}")


if __name__ == "__main__":
    main()