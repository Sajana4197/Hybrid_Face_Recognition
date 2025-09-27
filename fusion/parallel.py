
import numpy as np
from typing import Dict, List, Tuple
from common.hamming import hamming_distance_bits

def best_neuralhash_distance(query_bits: np.ndarray, nh_db: List[Dict]) -> Dict[str, int]:
    """Return best (min) Hamming distance per person for 96-bit NeuralHash."""
    per_person = {}
    for rec in nh_db:
        pid = rec["person_id"]
        best = 10**9
        for hv in rec.get("hashes", []):
            bits = np.array(hv, dtype=np.uint8)
            d = hamming_distance_bits(query_bits, bits)
            if d < best:
                best = d
        per_person[pid] = best if best < 10**9 else None
    return per_person

def best_hdic_distance(query_hv_bits: np.ndarray, hdic_db: List[Dict]) -> Dict[str, int]:
    """Return best (min) HDC Hamming distance per person for 10k-bit prototype(s)."""
    per_person = {}
    for rec in hdic_db:
        pid = rec["person_id"]
        best = 10**9
        for _, proto in rec.get("prototypes", {}).items():
            proto_bits = np.array(proto, dtype=np.uint8)
            d = hamming_distance_bits(query_hv_bits, proto_bits)
            if d < best:
                best = d
        per_person[pid] = best if best < 10**9 else None
    return per_person

def normalize_similarity_from_hamming(d: float, n_bits: int, clamp: bool = True) -> float:
    """Convert Hamming distance to similarity in [0,1]."""
    if d is None:
        return 0.0
    s = 1.0 - (float(d) / float(n_bits))
    if clamp:
        s = max(0.0, min(1.0, s))
    return s

def fuse_scores(nh_sim: float, hdic_sim: float, w_nh: float = 0.5, w_hdic: float = 0.5) -> float:
    w_sum = w_nh + w_hdic
    if w_sum <= 0:
        return 0.0
    return (w_nh * nh_sim + w_hdic * hdic_sim) / w_sum

def decide_parallel(
    nh_dists: Dict[str, int],
    hdic_dists: Dict[str, int],
    Tnh_reject: int,
    Thdic_accept: int,
    w_nh: float = 0.5,
    w_hdic: float = 0.5,
    fused_threshold: float = 0.7,
    require_both_modalities: bool = True,
) -> Tuple[str, Dict[str, float]]:
    """
    Make a decision in parallel mode.
    Returns (best_person_or_None, debug_info_per_pid)
    debug_info_per_pid maps pid -> dict with nh_d, hdic_d, nh_sim, hdic_sim, fused
    """
    debug = {}
    best_pid = None
    best_fused = -1.0
    for pid in sorted(set(list(nh_dists.keys()) + list(hdic_dists.keys()))):
        d_nh = nh_dists.get(pid, None)
        d_hdic = hdic_dists.get(pid, None)

        nh_sim = normalize_similarity_from_hamming(d_nh, 96)
        hdic_sim = normalize_similarity_from_hamming(d_hdic, 10000)

        fused = fuse_scores(nh_sim, hdic_sim, w_nh, w_hdic)

        debug[pid] = {
            "nh_d": d_nh, "hdic_d": d_hdic,
            "nh_sim": nh_sim, "hdic_sim": hdic_sim, "fused": fused
        }

        # Optional per-modality constraints for open-set robustness
        pass_modalities = True
        if require_both_modalities:
            if d_nh is None or d_hdic is None:
                pass_modalities = False
            else:
                pass_modalities = (d_nh <= Tnh_reject) and (d_hdic <= Thdic_accept)

        if pass_modalities and fused > best_fused:
            best_fused = fused
            best_pid = pid

    # Final threshold on fused score
    if best_pid is None:
        return None, debug
    if best_fused >= fused_threshold:
        return best_pid, debug
    return None, debug
