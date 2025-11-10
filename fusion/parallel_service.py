# fusion/parallel_service.py
from pathlib import Path
import numpy as np
import cv2
import json

# --- your modules ---
from preprocess.align import align_from_array           # array-based align (fast)
from neuralhash.adapter import compute_hash_bits        # -> (96,) {0,1}
from hdic.adapter import encode_hv                      # -> (10000,) {0,1}

# --- packed-flow utils ---
from fusion.bitpack import pack_probe_uint64, hamming_rows_all
from db.packed_store import PackedStore

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "db"

# ---------------------------------------------------------
# LEGACY JSONL LOADER (fallback only)
# ---------------------------------------------------------
def _load_watchlists_jsonl_arrays(repo_root: Path):
    """
    Legacy JSONL watchlist loader.
    Returns:
      nh_map: {pid: np.uint8 (N_hashes, 96)}
      hd_map: {pid: np.uint8 (N_clusters, 10000)}
    """
    nh_file = repo_root / "db" / "watchlist_neuralhash.jsonl"
    hdic_file = repo_root / "db" / "watchlist_hdic.jsonl"
    nh_map, hd_map = {}, {}

    if nh_file.exists():
        with nh_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
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
                if not s:
                    continue
                rec = json.loads(s)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                protos = (rec.get("prototypes") or {})
                if pid and protos:
                    mat = np.asarray(list(protos.values()), dtype=np.uint8).reshape(-1, 10000)
                    hd_map[pid] = mat

    return nh_map, hd_map

# ---------------------------------------------------------
# PACKED MODE: load once and align NH/HDIC person order
# ---------------------------------------------------------
def _try_load_packed():
    """Return packed arrays if available, else None to signal fallback."""
    try:
        nh_store = PackedStore(DB_DIR / "nh_packed", bits=96)
        hd_store = PackedStore(DB_DIR / "hdic_packed", bits=10000)

        NH_ROWS, NH_OFFS, NH_PIDS = nh_store.load_memmap()   # (R_nh, 2), (P_nh+1,), list[str]
        HD_ROWS, HD_OFFS, HD_PIDS = hd_store.load_memmap()   # (R_hd, 157), (P_hd+1,), list[str]

        # Common person order (intersection), keep NH order deterministically
        hd_pid_to_idx = {pid: i for i, pid in enumerate(HD_PIDS)}
        common_pids = [pid for pid in NH_PIDS if pid in hd_pid_to_idx]
        if not common_pids:
            return None

        nh_pid_to_idx = {pid: i for i, pid in enumerate(NH_PIDS)}
        NH_IDX = np.asarray([nh_pid_to_idx[pid] for pid in common_pids], dtype=np.int32)
        HD_IDX = np.asarray([hd_pid_to_idx[pid] for pid in common_pids], dtype=np.int32)

        return {
            "NH_ROWS": NH_ROWS,
            "NH_OFFS": NH_OFFS,
            "NH_PIDS": NH_PIDS,
            "HD_ROWS": HD_ROWS,
            "HD_OFFS": HD_OFFS,
            "HD_PIDS": HD_PIDS,
            "COMMON_PIDS": tuple(common_pids),
            "NH_IDX": NH_IDX,
            "HD_IDX": HD_IDX,
        }
    except Exception:
        return None

_PACKED = _try_load_packed()

# Legacy fallback only if packed is unavailable
if _PACKED is None:
    NH_MAP, HD_MAP = _load_watchlists_jsonl_arrays(REPO_ROOT)
    PERSON_IDS_LEGACY = tuple(sorted(set(NH_MAP.keys()) & set(HD_MAP.keys())))

# ---------------------------------------------------------
# UTIL: segment-wise minimum
# ---------------------------------------------------------
def _reduce_min_per_person(dists: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    dists:   (R,) distances across all rows
    offsets: (P+1,) start indices; slice i is [offsets[i], offsets[i+1])
    returns: (P,) per-person min
    """
    return np.minimum.reduceat(dists, offsets[:-1])

# ---------------------------------------------------------
# MAIN MATCH FUNCTION
# ---------------------------------------------------------
def match_frame(
    frame_bgr: np.ndarray,
    Tnh: float = 30,
    Thdic: float = 3100,
    w_nh: float = 0.4,
    w_hdic: float = 0.6,
    fused_th: float = 0.75,
    verbose: bool = False,
):
    # 1) Align
    face_rgb = align_from_array(frame_bgr, output_size=(160, 160), normalize=False)
    if face_rgb is None:
        return {"decision": "NO_FACE", "person_id": None, "scores": {}}

    # 2) Compute transient hashes (no embeddings stored)
    try:
        probe_nh_u8 = compute_hash_bits(face_rgb).astype(np.uint8).reshape(-1)  # (96,)
        probe_hv_u8 = encode_hv(face_rgb).astype(np.uint8).reshape(-1)          # (10000,)
    except Exception as e:
        return {"decision": "ERROR", "error": f"Encoding failed: {e}", "person_id": None, "scores": {}}

    # -------------------------
    # PACKED FAST PATH (preferred)
    # -------------------------
    if _PACKED is not None:
        NH_ROWS = _PACKED["NH_ROWS"]; NH_OFFS = _PACKED["NH_OFFS"]
        HD_ROWS = _PACKED["HD_ROWS"]; HD_OFFS = _PACKED["HD_OFFS"]
        COMMON  = _PACKED["COMMON_PIDS"]
        NH_IDX  = _PACKED["NH_IDX"];  HD_IDX  = _PACKED["HD_IDX"]

        if len(COMMON) == 0 or NH_ROWS.shape[0] == 0 or HD_ROWS.shape[0] == 0:
            return {"decision": "ERROR", "error": "Empty watchlist", "person_id": None, "scores": {}}

        # Pack probe once
        probe_nh_u64 = pack_probe_uint64(probe_nh_u8)   # (2,)
        probe_hv_u64 = pack_probe_uint64(probe_hv_u8)   # (157,)

        # Bulk distances to all rows
        d_all_nh = hamming_rows_all(probe_nh_u64, NH_ROWS)  # (R_nh,)
        d_all_hd = hamming_rows_all(probe_hv_u64, HD_ROWS)  # (R_hd,)

        # Per-person mins in each native order
        dmin_nh_all = _reduce_min_per_person(d_all_nh, NH_OFFS)  # (P_nh,)
        dmin_hd_all = _reduce_min_per_person(d_all_hd, HD_OFFS)  # (P_hd,)

        # Extract in COMMON person order
        d_nh = dmin_nh_all[NH_IDX]   # (P_common,)
        d_hd = dmin_hd_all[HD_IDX]   # (P_common,)

        # Fuse & pick best
        Snh        = 1.0 - (d_nh / 96.0)
        Shdic_norm = 1.0 - (d_hd / 10000.0)
        Sfinal     = (w_nh * Snh) + (w_hdic * Shdic_norm)

        best_idx = int(np.argmax(Sfinal))
        best_pid = COMMON[best_idx]
        best = {
            "d_nh":       int(d_nh[best_idx]),
            "d_hdic":     int(d_hd[best_idx]),
            "Snh":        float(Snh[best_idx]),
            "Shdic_norm": float(Shdic_norm[best_idx]),
            "Sfinal":     float(Sfinal[best_idx]),
        }

        is_match = (best["d_nh"] < Tnh) and (best["d_hdic"] < Thdic) and (best["Sfinal"] >= fused_th)
        return {"decision": "MATCH" if is_match else "NO_MATCH", "person_id": best_pid, "scores": best}

    # -------------------------
    # LEGACY JSONL FALLBACK (slower; for compatibility)
    # -------------------------
    if not PERSON_IDS_LEGACY:
        return {"decision": "ERROR", "error": "Empty watchlist", "person_id": None, "scores": {}}

    best_pid, best_sfinal, best_metrics = None, -1.0, None
    for pid in PERSON_IDS_LEGACY:
        hashes_mat = NH_MAP.get(pid)
        protos_mat = HD_MAP.get(pid)
        if hashes_mat is None or protos_mat is None:
            continue

        d_nh = int(np.min(np.sum(hashes_mat != probe_nh_u8, axis=1))) if hashes_mat.size else 96
        d_hdic = int(np.min(np.sum(protos_mat != probe_hv_u8, axis=1))) if protos_mat.size else 10_000

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

        if (d_nh < Tnh and d_hdic < Thdic and Sfinal >= 0.95):
            break

    if best_pid is None:
        return {"decision": "NO_MATCH", "person_id": None, "scores": {}}

    is_match = (
        (best_metrics["d_nh"] < Tnh)
        and (best_metrics["d_hdic"] < Thdic)
        and (best_metrics["Sfinal"] >= fused_th)
    )
    return {"decision": "MATCH" if is_match else "NO_MATCH", "person_id": best_pid, "scores": best_metrics}
