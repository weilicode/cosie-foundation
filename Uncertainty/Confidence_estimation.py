#!/usr/bin/env python

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  


#: The pre-built calibration, beside this script.
PREBUILT = Path(__file__).resolve().parent / "Confidence_calibration"

#: Confidence is 1 at P_MIN and 0 at P_MAX, linear in -log10(p) between.
P_MIN, P_MAX = 0.05, 0.5


#: Top of the prediction's colour range, as a percentile of the section's values.
OWN_TOP = 99.9

#: Painted where the section has no unit: white on the prediction, since turbo
#: ends in dark red; mid grey on the confidence, since `hot` ends in white.
EMPTY = (255, 255, 255)
CONFIDENCE_EMPTY = (172, 172, 172)


# --------------------------------------------------------------------------- #
# .h5ad access
# --------------------------------------------------------------------------- #
def _index(handle, group: str) -> np.ndarray:
    node = handle[group]
    key = node.attrs.get("_index", "_index")
    return node[key.decode() if isinstance(key, bytes) else str(key)][:]


def var_names(path) -> List[str]:
    with h5py.File(path, "r") as handle:
        return [name.decode() if isinstance(name, bytes) else str(name)
                for name in _index(handle, "var")]


def n_obs(path) -> int:
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        return int(x.shape[0] if isinstance(x, h5py.Dataset) else x.attrs["shape"][0])


def read_columns(path, names: Sequence[str], block: int = 65536) -> np.ndarray:
    """``(n_obs, len(names))`` float32 values of the named variables.

    Read in row blocks: on an unchunked dataset each block is one contiguous
    read, where selecting columns from the file would re-read it per column.
    """
    lookup = {name: column for column, name in enumerate(var_names(path))}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise KeyError(f"{Path(path).name} has no variable(s) {missing}")
    columns = np.array([lookup[name] for name in names], dtype=np.int64)
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        if not isinstance(x, h5py.Dataset):
            raise ValueError(f"{Path(path).name}: X is sparse; save the predictions dense")
        out = np.empty((x.shape[0], len(columns)), dtype=np.float32)
        for start in range(0, x.shape[0], block):
            stop = min(start + block, x.shape[0])
            out[start:stop] = x[start:stop][:, columns]
    return out


def read_positions(path: Path, n: int) -> np.ndarray:
    """``(n, 2)`` integer bin positions from ``obsm/spatial``."""
    with h5py.File(path, "r") as handle:
        node = handle.get("obsm/spatial")
        if not isinstance(node, h5py.Dataset) or node.ndim != 2 or node.shape[0] != n:
            raise ValueError(f"{path.name} has no obsm/spatial for its {n} units")
        return np.rint(node[:, :2]).astype(np.int64)


# --------------------------------------------------------------------------- #
# The pre-built calibration
# --------------------------------------------------------------------------- #
class Prebuilt:
    """One modality's ``calibration.npy`` and ``meta.json``.

    ``calibration.npy[c, k, i]`` is the p-value of a unit of cluster ``k``
    predicted at ``grid[i]`` for channel ``c``, where
    ``grid = linspace(0, grid_top, n_grid)``.
    """

    def __init__(self, directory: Path) -> None:
        with open(directory / "meta.json") as handle:
            self.meta = json.load(handle)
        #: Row ``i`` of ``calibration.npy`` is ``entries[i]``.
        self.entries: List[Dict] = self.meta["channels"]
        self.channels = [entry["name"] for entry in self.entries]
        #: Memory-mapped, so a target costs only its own (clusters, grid) slice.
        self.table = np.load(directory / "calibration.npy", mmap_mode="r")

    def grid(self, row: int) -> np.ndarray:
        return np.linspace(0.0, float(self.entries[row]["grid_top"]), self.table.shape[2])



def resolve_targets(requested: Sequence[str], channels: Sequence[str], available: Dict[str, str]) -> List[Tuple[str, str]]:
    calibration_lookup = {name.upper(): name for name in channels}
    prediction_lookup = {name.upper(): stored for name, stored in available.items()}
    resolved = []

    for name in requested:
        calibration_name = calibration_lookup.get(name.upper())
        prediction_name = prediction_lookup.get(name.upper())

        if calibration_name is None:
            print(f"[utopia] {name} not found in calibration data", flush=True)
            continue
        if prediction_name is None:
            print(f"[utopia] {name} not found in prediction file", flush=True)
            continue

        pair = (calibration_name, prediction_name)
        if pair not in resolved:
            resolved.append(pair)

    return resolved

def read_clusters(path: Path, n: int, meta: Dict) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        node = handle.get("obs/assigned_label")
        if node is None:
            raise ValueError(f"{path.name} has no obs['assigned_label']")

        if isinstance(node, h5py.Dataset):
            labels = node[:]
        elif isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
            codes = node["codes"][:]
            categories = node["categories"][:]
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
            labels = np.array([categories[x] if x >= 0 else -1 for x in codes])
        else:
            raise ValueError(f"{path.name}: unsupported obs['assigned_label'] format")

    labels = np.asarray(labels).astype(np.int64)
    if labels.shape != (n,):
        raise ValueError(f"{path.name}: {labels.shape} assigned labels for {n} units")

    lookup = np.asarray(meta["from_he_cluster"], dtype=np.int64)
    inside = (labels >= 0) & (labels < len(lookup))

    if not inside.all():
        print(f"[utopia] WARNING: {int((~inside).sum()):,} units carry a label outside 0-{len(lookup)-1}; they are left at p = 1", flush=True)

    return np.where(inside, lookup[np.clip(labels, 0, len(lookup)-1)], -1)


def modality_variables(names: Sequence[str], kind: str) -> Dict[str, str]:
    if kind == "rna":
        return {name: name for name in names if not name.startswith("protein_")}
    return {name[len("protein_"):]: name for name in names if name.startswith("protein_")}


# --------------------------------------------------------------------------- #
# P-values
# --------------------------------------------------------------------------- #
def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    count = len(pvals)
    if count == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(pvals)
    scaled = count * np.asarray(pvals, dtype=float)[order] / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def batch_tolerance(test: np.ndarray, null_fraction: float,
                    calib_lower: Optional[Sequence[float]], levels: np.ndarray) -> float:
    if calib_lower is None:
        return 0.0
    lower = test[test <= np.quantile(test, null_fraction)]
    if len(lower) == 0:
        return 0.0
    shifts = np.quantile(lower, levels) - np.asarray(calib_lower, dtype=float)
    return -float(np.max(np.clip(shifts, 0.0, None)))


def cluster_pvalues(prediction: np.ndarray, clusters: np.ndarray, prebuilt: Prebuilt,
                    row: int) -> Tuple[np.ndarray, np.ndarray]:
    entry = prebuilt.entries[row]
    grid = prebuilt.grid(row)
    table = np.asarray(prebuilt.table[row], dtype=float)
    levels = np.asarray(prebuilt.meta["quantile_levels"], dtype=float)

    pvals = np.ones(len(prediction), dtype=float)
    adjusted = np.ones(len(prediction), dtype=float)
    for cluster in range(table.shape[0]):
        rows = np.flatnonzero(clusters == cluster)
        if not len(rows):
            continue
        values = prediction[rows].astype(np.float64)
        shift = batch_tolerance(values, entry["null_fraction"][cluster],
                                entry["lower_quantiles"][cluster], levels)
        index = np.searchsorted(grid, values + shift, side="right") - 1
        raw = table[cluster][np.maximum(index, 0)]
        pvals[rows] = raw
        adjusted[rows] = bh_adjust(raw)
    return pvals, adjusted


def to_confidence(pvals: np.ndarray) -> np.ndarray:
    floored = np.maximum(np.asarray(pvals, dtype=float), np.finfo(float).tiny)
    scores = (-np.log10(floored) + np.log10(P_MAX)) / (-np.log10(P_MIN) + np.log10(P_MAX))
    return np.clip(scores, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #
def raster(values: np.ndarray, spots: np.ndarray) -> np.ndarray:
    rows = spots[:, 0] - spots[:, 0].min()
    columns = spots[:, 1] - spots[:, 1].min()
    image = np.full((int(rows.max()) + 1, int(columns.max()) + 1), np.nan, dtype=np.float32)
    image[rows, columns] = values
    return image


def colourise(image: np.ndarray, low: float, high: float, colormap: str,
              empty: Sequence[int]) -> np.ndarray:
    drawn = np.isfinite(image)
    scaled = np.clip((image[drawn] - low) / (high - low if high > low else 1.0), 0.0, 1.0)
    out = np.empty((*image.shape, 3), dtype=np.uint8)
    out[:] = np.asarray(empty, dtype=np.uint8)
    out[drawn] = (matplotlib.colormaps[colormap](scaled)[:, :3] * 255).astype(np.uint8)
    return out


def own_top(values: np.ndarray) -> float:
    high = float(np.percentile(values, OWN_TOP))
    if not high > 0:
        high = float(np.max(values))
    return high if high > 0 else 1e-6


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UTOPIA prediction and confidence maps.")
    parser.add_argument("--prediction", required=True,
                        help="predicted .h5ad of (X dense, obsm/spatial)")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--rna", action="store_true", help="use the RNA calibration")
    kind.add_argument("--protein", action="store_true", help="use the protein calibration")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="genes or proteins, space or comma separated")
    parser.add_argument("--out", help="output folder")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    path = Path(args.prediction).expanduser().resolve()
    kind = "rna" if args.rna else "protein"
    prebuilt = Prebuilt(PREBUILT / kind)

    names = var_names(path)
    available = modality_variables(names, kind)
    overlap = len(set(name.upper() for name in available) & set(name.upper() for name in prebuilt.channels))
    if overlap < 0.25 * max(len(available), len(prebuilt.channels)):
        print(f"[utopia] WARNING: {path.name} shares only {overlap} variables with the {kind} calibration; is --{kind} correct?", flush=True)
    
    requested = [name for value in args.targets for name in re.split(r"[,\s]+", value) if name]
    targets = resolve_targets(requested, prebuilt.channels, available)
    if not targets:
        return
    
    n = n_obs(path)
    positions = read_positions(path, n)
    
    inferred_path = path.parent / "adata_query_inferred.h5ad"
    if not inferred_path.exists():
        raise FileNotFoundError(f"{inferred_path} not found. Please complete COSIE-Foundation label transfer first.")
    
    clusters = read_clusters(inferred_path, n, prebuilt.meta)
    
    stored_names = [stored_name for _, stored_name in targets]
    values = read_columns(path, stored_names)
    
    out = Path(args.out).expanduser() if args.out else path.parent / "Confidence"
    print(f"[utopia] {path.name}: {n:,} units, {kind} calibration", flush=True)
    for column, (target, _) in enumerate(targets):
        prediction = values[:, column]
        _, adjusted = cluster_pvalues(prediction, clusters, prebuilt, prebuilt.channels.index(target))
        confidence = to_confidence(adjusted)
    
        directory = out / f"{target}_{kind}"
        directory.mkdir(parents=True, exist_ok=True)
    
        np.savez_compressed(directory / "confidence_results.npz", confidence=confidence.astype(np.float32))
    
        high = own_top(prediction)
        pred_rgb = colourise(raster(prediction, positions), 0.0, high, "turbo", EMPTY)
        conf_rgb = colourise(raster(confidence, positions), 0.0, 1.0, "hot", CONFIDENCE_EMPTY)
    
        plt.imsave(directory / "prediction.png", pred_rgb)
        plt.imsave(directory / "confidence.png", conf_rgb)
    
        print(f"[utopia] {target} -> {directory}", flush=True)




if __name__ == "__main__":
    main()
