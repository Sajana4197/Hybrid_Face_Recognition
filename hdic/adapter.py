import numpy as np
from .feature_extractor import generate_embedding2
from .encode_hv import encode_embedding_to_hv

def extract_embedding_impl(aligned_rgb_image: np.ndarray) -> np.ndarray:
    """
    Takes an aligned RGB numpy array (160x160x3),
    returns a 512D FaceNet embedding (float32).
    """
    # Ensure proper dtype
    if aligned_rgb_image.dtype != np.uint8:
        aligned_rgb_image = (aligned_rgb_image * 255).astype(np.uint8) if aligned_rgb_image.max() <= 1.0 else aligned_rgb_image.astype(np.uint8)

    emb = generate_embedding2(aligned_rgb_image)  # (512,)
    return np.array(emb, dtype=np.float32)

def encode_hv_impl(aligned_rgb_image: np.ndarray) -> np.ndarray:
    """
    Takes an aligned RGB numpy array (160x160x3),
    returns a 10,000D binary hypervector (uint8 0/1).
    """
    emb = extract_embedding_impl(aligned_rgb_image)  # (512,)
    hv = encode_embedding_to_hv(emb)                # (10000,)
    return hv.astype(np.uint8)

# Wrappers for the hybrid pipeline
def extract_embedding(aligned_rgb_image):
    return extract_embedding_impl(aligned_rgb_image)

def encode_hv(aligned_rgb_image):
    return encode_hv_impl(aligned_rgb_image)
