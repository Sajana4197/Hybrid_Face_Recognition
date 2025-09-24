import base64, numpy as np
from sklearn.cluster import KMeans
from common.hamming import bundle_majority

def build_cluster_prototypes(hypervectors, num_clusters=3, flip_p=0.0):
    """
    Build cluster prototypes from a list of 10,000D hypervectors.
    """
    
    hypervectors = np.stack(hypervectors)
    # Cluster on embeddings space (optional: use raw hypervectors, or 512D embeddings if desired)
    kmeans = KMeans(n_clusters=min(num_clusters, len(hypervectors)), n_init=5, random_state=42)
    labels = kmeans.fit_predict(hypervectors)

    prototypes = {}
    for c in sorted(set(labels)):
        idxs = np.where(labels == c)[0]
        cluster_hvs = hypervectors[idxs]
        proto = bundle_majority(cluster_hvs)
        prototypes[f"cluster_{c}"] = proto
    return prototypes


def bits_to_base64(bits: np.ndarray) -> str:
    return base64.b64encode(np.packbits(bits).tobytes()).decode('ascii')

def base64_to_bits(s: str, bit_dim:int) -> np.ndarray:
    arr=np.frombuffer(base64.b64decode(s),dtype=np.uint8)
    bits=np.unpackbits(arr)[:bit_dim]
    return bits.astype(np.uint8)
