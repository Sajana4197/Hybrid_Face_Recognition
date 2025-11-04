# fusion/parallel_service.py
from pathlib import Path
import cv2, numpy as np
import json

# --- Use your real modules ---
from preprocess.align import align_from_array           # array-based align (fast)
from neuralhash.adapter import compute_hash_bits        # 96-bit NH vector
from hdic.encode_hv import encode_embedding_to_hv       # returns 10k-D HV directly
from hdic.feature_extractor import generate_embedding2  # import your 512-D embedder

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root

def _load_watchlists_jsonl_arrays(repo_root: Path):
    """
    Load both NH and HDIC from JSONL and convert to compact numpy arrays once.
    Returns:
      nh_map: {pid: np.uint8 shape (N_hashes, 96)}
      hd_map: {pid: np.uint8 shape (N_clusters, 10000)}
    """
    nh_file = repo_root / "db" / "watchlist_neuralhash.jsonl"
    hdic_file = repo_root / "db" / "watchlist_hdic.jsonl"
    nh_map, hd_map = {}, {}

    if nh_file.exists():
        with nh_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                rec = json.loads(s)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                hashes = rec.get("hashes", [])
                if pid and hashes:
                    arr = np.asarray(hashes, dtype=np.uint8).reshape(-1, 96)
                    nh_map[pid] = arr

    if hdic_file.exists():
        with hdic_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                rec = json.loads(s)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                protos = (rec.get("prototypes") or {})
                if pid and protos:
                    mat = np.asarray(list(protos.values()), dtype=np.uint8).reshape(-1, 10000)
                    hd_map[pid] = mat

    return nh_map, hd_map

# Load once (cached for all requests)
NH_MAP, HD_MAP = _load_watchlists_jsonl_arrays(REPO_ROOT)
PERSON_IDS = tuple(sorted(set(NH_MAP.keys()) & set(HD_MAP.keys())))

# ---------------------------------------------------------
# FAST DISTANCES
# ---------------------------------------------------------
def _nh_min_distance(probe_bits: np.ndarray, hashes_mat: np.ndarray) -> int:
    """
    Vectorized Hamming distance to a person's NH hashes
    probe_bits: (96,) uint8 in {0,1}
    hashes_mat: (N,96) uint8 in {0,1}
    """
    if hashes_mat is None or hashes_mat.size == 0:
        return 96
    # XOR and count mismatches (since 0/1) == !=
    return int(np.min(np.sum(hashes_mat != probe_bits, axis=1)))

def _hdic_min_distance(probe_hv: np.ndarray, protos_mat: np.ndarray) -> int:
    """
    Vectorized Hamming distance to a person's HDIC prototype hypervectors (binary).
    probe_hv: (10000,) uint8 in {0,1}
    protos_mat: (K,10000) uint8 in {0,1}
    """
    if protos_mat is None or protos_mat.size == 0:
        return 10_000
    return int(np.min(np.sum(protos_mat != probe_hv, axis=1)))

# ---------------------------------------------------------
# MAIN MATCH FUNCTION (minimal logging, fast path)
# ---------------------------------------------------------
def match_frame(
    frame_bgr: np.ndarray,
    Tnh: float = 30,
    Thdic: float = 3100,
    w_nh: float = 0.4,
    w_hdic: float = 0.6,
    fused_th: float = 0.75,
    verbose: bool = False,  # keep logs minimal by default
):
    # Align face (array-based; no file I/O)
    face_rgb = align_from_array(frame_bgr, output_size=(160,160), normalize=False)
    if face_rgb is None:
        return {"decision": "NO_FACE", "person_id": None, "scores": {}}

    # Feature extraction
    try:
        probe_nh = compute_hash_bits(face_rgb).astype(np.uint8).reshape(-1)
        emb = generate_embedding2(face_rgb)
        probe_hv = encode_embedding_to_hv(emb).astype(np.uint8).reshape(-1)
    except Exception as e:
        return {"decision": "ERROR", "error": f"Encoding failed: {e}", "person_id": None, "scores": {}}

    if len(PERSON_IDS) == 0:
        return {"decision": "ERROR", "error": "Empty watchlist", "person_id": None, "scores": {}}

    # Score each person
    best_pid, best_sfinal, best_metrics = None, -1.0, None
    for pid in PERSON_IDS:
        d_nh = _nh_min_distance(probe_nh, NH_MAP.get(pid))
        d_hdic = _hdic_min_distance(probe_hv, HD_MAP.get(pid))
        Snh = 1.0 - (d_nh / 96.0)
        Shdic_norm = 1.0 - (d_hdic / 10000.0)
        Sfinal = (w_nh * Snh) + (w_hdic * Shdic_norm)

        if Sfinal > best_sfinal:
            best_sfinal = Sfinal
            best_pid = pid
            best_metrics = {
                "d_nh": d_nh,
                "d_hdic": d_hdic,
                "Snh": Snh,
                "Shdic_norm": Shdic_norm,
                "Sfinal": Sfinal,
            }

        # Early exit for very strong matches to save time
        if (d_nh < Tnh and d_hdic < Thdic and Sfinal >= 0.95):
            break

    if best_pid is None:
        return {"decision": "NO_MATCH", "person_id": None, "scores": {}}

    is_match = (
        (best_metrics["d_nh"] < Tnh)
        and (best_metrics["d_hdic"] < Thdic)
        and (best_metrics["Sfinal"] >= fused_th)
    )
    decision = "MATCH" if is_match else "NO_MATCH"

    return {"decision": decision, "person_id": best_pid, "scores": best_metrics}