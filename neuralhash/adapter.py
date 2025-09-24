import os
import numpy as np
from .neuralhash_core import (
    load_pca,
    load_hyperplanes,
    get_96d_neural_hash_from_array,
)

# ---------------- Load global assets ---------------- #

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
PCA_PATH = os.path.join(ASSETS_DIR, "pca_512_to_128.pkl")
DAT_PATH = os.path.join(ASSETS_DIR, "neuralhash_128x96_seed1.dat")

if not os.path.exists(PCA_PATH):
    raise FileNotFoundError(f"PCA file not found: {PCA_PATH}")
if not os.path.exists(DAT_PATH):
    raise FileNotFoundError(f"NeuralHash .dat file not found: {DAT_PATH}")

PCA_MODEL = load_pca(PCA_PATH)
HYPERPLANES = load_hyperplanes(DAT_PATH)

# ---------------- API functions ---------------- #

def compute_hash_bits(rgb_image: np.ndarray) -> np.ndarray:
    """
    Compute the 96-bit NeuralHash for an aligned RGB face image.

    Input:
        rgb_image: numpy array (160x160x3, dtype=uint8 or float in [0,1])
    Output:
        numpy array of shape (96,), dtype=uint8 (0/1 bits)
    """
    hbits = get_96d_neural_hash_from_array(rgb_image, PCA_MODEL, HYPERPLANES)
    if hbits.shape[0] != 96:
        raise ValueError(f"Unexpected NeuralHash output shape: {hbits.shape}")
    return hbits
