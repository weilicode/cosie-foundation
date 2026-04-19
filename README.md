

<p align="left">
  <img src=./Image/COSIE_Foundation_logo.png width="300"/>
</p>

[![python >3.9.19](https://img.shields.io/badge/python-3.9.19-blue)](https://www.python.org/) 



**COSIE-Foundation** is a unified framework for large-scale spatial multimodal integration, label transfer, and virtual molecular prediction. It supports two main use cases:

**1. Direct inference** using a pretrained COSIE-Foundation model for query section, including
   - Label transfer
   - Virtual gene / protein prediction

**2. Training your own model from scratch**  
   - Train COSIE-Foundation on your own ultra large-scale spatial multimodal dataset.




<p align="center">
  <img width="100%" src=./Image/COSIE_Foundation_framework.png>
</p>


# Installation




For convenience, we recommend creating and activating a dedicated conda environment before installing COSIE-Foundation.
If you haven't installed conda yet, we suggest using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), a lightweight distribution of conda.



```bash
conda create -n cosie_foundation_env python=3.9.19
conda activate cosie_foundation_env
```      

The COSIE package can be downloaded by:
```bash
git clone https://github.com/weilicode/cosie-foundation.git
cd cosie-foundation
```


The cosie_foundation_env environment can be used in jupyter notebook by:

```bash
pip install ipykernel
python -m ipykernel install --user --name=cosie_foundation_env
```




COSIE-Foundation is built upon [![torch-2.4.0](https://img.shields.io/badge/torch-2.4.0-orange)](https://pytorch.org/) and [![torch__geometric-2.5.3](https://img.shields.io/badge/torch__geometric-2.5.3-blueviolet)](https://pytorch-geometric.readthedocs.io/en/latest/). Using GPU acceleration can significantly speed up the training process. If you plan to use a GPU, please make sure that PyTorch and PyTorch Geometric are installed with versions that are compatible with your local CUDA version. For example, if you are using CUDA 12.1, you can install the required packages as follows:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch_geometric==2.5.3
```

All other required packages are listed in [requirements.txt](requirements.txt). You can install them by running:

```bash
pip install -r requirements.txt
```



# 🔹 1. Direct inference using pretrained COSIE-Foundation

Copy the core package into your inference folder:

```
cp -r ./COSIE_Foundation ./Inference/
cd Inference
```



## 1.1 Label transfer

Given a query section, COSIE-Foundation projects it into COSIE embedding space and outputs fine-grained tissue structure annotations.

- Download the pretrained COSIE-Foundation checkpoint from
[COSIE_Foundation_checkpoint.zip](https://upenn.box.com/s/60vz0bnigt38y7332mpxfvkiam067zt2). Then unzip and place it under: `<inference-root>/COSIE_Foundation_checkpoint/`
- Prepare query data `adata_query.h5ad`, which must contain:
    - X: feature matrix  
    - obsm["spatial"]

    Example query data can be downloaded from [Here](chage link)

- Run label transfer

    ```
    python 1_label_transfer.py \
        --out-root /path/to/inference-root \
        --adata-path /path/to/adata_query.h5ad
    ```

    The following outputs will be saved in `inference-root`:

    - adata_query_inferred.h5ad — inferred embeddings and labels
    - celltype_labels.png — visualization of predicted structures



## 1.2. Virtual prediction

Given the inferred COSIE embeddings from label transfer, this step predicts virtual RNA / Protein features for the query section.

- Download the virtual prediction reference from [Virtual_prediction_reference.zip](https://upenn.box.com/s/60vz0bnigt38y7332mpxfvkiam067zt2). Then unzip and place it under: `<inference-root>/Virtual_prediction_reference/`

- Make sure the previous label transfer has been completed and the following file exists:

    - `<inference-root>/adata_query_inferred.h5ad`

- Run virtual prediction

    ```
    python Step4_2_virtual_prediction.py \
        --inference-root /path/to/inference-root \
        --bundle-dir /path/to/inference-root/Virtual_prediction_reference
    ```

    The following output will be saved in `inference-root`:

    - `adata_query_predicted.h5ad` — predicted RNA / Protein features



## 🔸 2. Train your own model

To train COSIE-Foundation on your own dataset:

```
cp -r ./COSIE_Foundation ./Train/
cd Train
```


## 2.1. Data preprocessing (Optional)

This step prepares HE, RNA, and Protein data for COSIE-Foundation training. You can go directly to Step 2 if your data are already preprocessed.

- Organize your input data as:
    ```
    Data/
    ├── HE/
    │   ├── adata_s1.h5ad
    │   ├── adata_s2.h5ad
    │   └── ...
    ├── RNA/
    │   ├── adata_s1.h5ad
    │   ├── adata_s2.h5ad
    │   └── ...
    └── Protein/
        ├── adata_s4.h5ad
        ├── adata_s5.h5ad
        └── ...
    ```
    Each file must follow `adata_<section_name>.h5ad`. Required contents of each .h5ad:
    - X: raw data matrix
    - obsm["spatial"]: spatial coordinates
    - var_names: feature names (for RNA / Protein)

- Run preprocessing
    ```
    python Preprocessing_HE.py --data-root /your_data_path
    python Preprocessing_RNA.py --data-root /your_data_path
    python Preprocessing_Protein.py --data-root /your_data_path
    python Summarize_all_modalities.py --data-root /your_data_path/Data_preprocessing
    ```

- Output file structure:
    ```
    Data_preprocessing/
    ├── feature_dict_concat.pkl
    ├── data_dict_processed_concat.pkl
    ├── spatial_loc_dict.pkl
    ```

## 2.2. Build your own dictionaries (skip preprocessing)

If your data already contain low-dimensional modality features, you can Step 1 and directly construct COSIE inputs. 

- Organize your data as follows:
    ```
    project_root/
    ├── sections.txt
    ├── HE/
    │   ├── adata_s1.h5ad
    │   ├── adata_s2.h5ad
    │   └── ...
    ├── RNA/
    │   ├── adata_s1.h5ad
    │   ├── adata_s2.h5ad
    │   └── ...
    └── Protein/
        ├── adata_s1.h5ad
        ├── adata_s3.h5ad
        └── ...
    ```
    Each file must contain:
    - obsm["spatial"]
    - modality-specific embeddings stored in obsm
    - sections.txt defines the section order with one section name per line:
        ```
        s1
        s2
        s3
        ...
        ```
- Run build_your_own_dict.py:
    ```
    python Build_your_own_dict.py --project-root /path/to/project_root
    ```

- Output file structure:
    ```
    project_root/Data_preprocessing/
    ├── feature_dict_concat.pkl
    ├── data_dict_processed_concat.pkl
    ├── spatial_loc_dict.pkl
    ```

## 2.3. Training and clustering

```
python Training.py --project-root /path/to/project_root
python Clustering.py --project-root /path/to/project_root --n-clusters 25
```

Embedding and clustering results will be saved in `/Embedding` and `/Clustering`.