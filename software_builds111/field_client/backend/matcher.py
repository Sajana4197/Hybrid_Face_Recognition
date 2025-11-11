import numpy as np
from typing import Tuple

def hamming_distance_bits(a: np.ndarray, b: np.ndarray) -> int:
    # a,b: uint8 arrays of 0/1, length 96
    return int(np.sum(a != b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    # HDIC distance (smaller is better)
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.dot(d, d)))

def score_person_distances(
    probe_nh: np.ndarray, probe_hv: np.ndarray, person_rec: dict
) -> Tuple[int, float, float, float]:
    """
    Returns:
      d_nh (min Hamming over hashes),
      d_hdic (min Euclid over prototypes),
      Snh (1 - d_nh/96),
      Shdic_norm (1 - d_hdic/10000)   # adjust denominator if your scale differs
    """
    # NH: min Hamming
    dists_nh = [hamming_distance_bits(probe_nh, h) for h in person_rec["nh_hashes"]]
    d_nh = min(dists_nh) if dists_nh else 96
    Snh = 1.0 - (d_nh / 96.0)

    # HDIC: min Euclidean across prototypes
    dists_hv = [euclidean_distance(probe_hv, hv) for hv in person_rec["hdic_prototypes"]]
    d_hdic = min(dists_hv) if dists_hv else 1e9
    # Normalize ONLY for fusion, do NOT use this for threshold gate (we use raw distance there)
    Shdic_norm = 1.0 - (d_hdic / 10000.0)

    return d_nh, d_hdic, Snh, Shdic_norm

def fuse_parallel(Snh: float, Shdic_norm: float, w_nh: float, w_hdic: float) -> float:
    return float(w_nh * Snh + w_hdic * Shdic_norm)
