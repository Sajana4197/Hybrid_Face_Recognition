import numpy as np

def hamming_distance_bits(a: np.ndarray, b: np.ndarray) -> int:
    if a.dtype != np.uint8: a = a.astype(np.uint8)
    if b.dtype != np.uint8: b = b.astype(np.uint8)
    return int(np.count_nonzero(a ^ b))

def bundle_majority(hvs: np.ndarray) -> np.ndarray:
    votes = hvs.sum(axis=0)
    return (votes * 2 >= hvs.shape[0]).astype(np.uint8)
