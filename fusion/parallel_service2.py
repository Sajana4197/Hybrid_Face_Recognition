# fusion/parallel_service.py
from pathlib import Path
import tempfile, os, cv2, numpy as np
import json

# --- Use your real modules ---
from preprocess.align import load_and_align          # aligns using MTCNN
from neuralhash.adapter import compute_hash_bits     # 96-bit NH vector
from hdic.encode_hv import encode_embedding_to_hv    # returns 10k-D HV directly
from hdic.feature_extractor import generate_embedding2   # import your 512-D embedder

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # points to repo root

def _load_watchlists_jsonl(repo_root: Path):
    """Load both NH and HDIC watchlists from db/ folder"""
    nh_file = repo_root / "db" / "watchlist_neuralhash.jsonl"
    hdic_file = repo_root / "db" / "watchlist_hdic.jsonl"
    nh_map, hd_map = {}, {}

    if nh_file.exists():
        with nh_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                nh_map[pid] = rec.get("hashes", [])

    if hdic_file.exists():
        with hdic_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                hd_map[pid] = list((rec.get("prototypes") or {}).values())

    return nh_map, hd_map


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _align_from_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Your aligner expects a file path, so we write temp JPEG."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, frame_bgr)
    try:
        face_rgb = load_and_align(tmp_path, output_size=(160, 160), normalize=False)
        return face_rgb
    finally:
        try: os.remove(tmp_path)
        except: pass


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def _nh_min_distance(probe_bits: np.ndarray, hashes_list: list[list[int]]) -> int:
    """Minimum Hamming distance across a person's stored NH hashes."""
    if not hashes_list:
        return 96
    pb = probe_bits.astype(np.uint8).reshape(-1)
    dmin = 96
    for h in hashes_list:
        ha = np.array(h, dtype=np.uint8).reshape(-1)
        d = _hamming_distance(pb, ha)
        if d < dmin:
            dmin = d
    return dmin


def _hdic_min_distance(probe_hv: np.ndarray, prototypes: list[list[int]]) -> float:
    """Minimum Hamming distance across HDIC prototypes (for 0/1 binary HVs)."""
    if not prototypes:
        return 1e9
    p = probe_hv.astype(np.uint8).reshape(-1)
    dmin = 1e9
    for proto in prototypes:
        q = np.array(proto, dtype=np.uint8).reshape(-1)
        # --- Real Hamming distance (count differing bits) ---
        d = int(np.sum(p != q))
        if d < dmin:
            dmin = d
    return float(dmin)



# ---------------------------------------------------------
# MAIN MATCH FUNCTION
# ---------------------------------------------------------
def match_frame(
    frame_bgr: np.ndarray,
    Tnh: float = 30,
    Thdic: float = 3100,
    w_nh: float = 0.5,
    w_hdic: float = 0.5,
    fused_th: float = 0.70,
):
    """
    Full parallel NH + HDIC pipeline for a webcam frame.
    Reuses your original logic from match_parallel.py.
    """

    # 1️⃣ Face alignment
    face_rgb = _align_from_frame(frame_bgr)
    if face_rgb is None:
        return {"decision": "NO_FACE", "person_id": None, "scores": {}}

    # 2️⃣ Feature extraction (real modules)
    try:
        print("[DEBUG] Align success:", face_rgb.shape)
        probe_nh = compute_hash_bits(face_rgb)
        print("[DEBUG] NH OK")

        embedding = generate_embedding2(face_rgb)  # 512-D FaceNet features
        print("[DEBUG] Embedding shape:", embedding.shape)

        probe_hv = encode_embedding_to_hv(embedding).astype(np.float32)  # 10k-D HDIC HV
        print("[DEBUG] HDIC OK")

    except Exception as e:
        print("[ERROR] Encoding failed:", e)
        return {
            "decision": "ERROR",
            "error": f"Encoding failed: {e}",
            "person_id": None,
            "scores": {},
        }

    # 3️⃣ Load both watchlists
    nh_map, hd_map = _load_watchlists_jsonl(REPO_ROOT)
    person_ids = sorted(set(nh_map.keys()) & set(hd_map.keys()))
    if not person_ids:
        return {"decision": "ERROR", "error": "Empty watchlist", "person_id": None, "scores": {}}

    # 4️⃣ Score each person
    best_pid, best_sfinal, best_metrics = None, -1.0, None
    for pid in person_ids:
        d_nh = _nh_min_distance(probe_nh, nh_map.get(pid, []))
        d_hdic = _hdic_min_distance(probe_hv, hd_map.get(pid, []))
        Snh = 1.0 - (d_nh / 96.0)
        Shdic_norm = 1.0 - (d_hdic / 10000.0)  # normalization for fusion
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

    if best_pid is None:
        return {"decision": "NO_MATCH", "person_id": None, "scores": {}}

    # 5️⃣ Apply your original triple-gate rule
    is_match = (
        (best_metrics["d_nh"] < Tnh)
        and (best_metrics["d_hdic"] < Thdic)
        and (best_metrics["Sfinal"] >= fused_th)
    )
    decision = "MATCH" if is_match else "NO_MATCH"

    return {"decision": decision, "person_id": best_pid, "scores": best_metrics}


def get_nh_and_hdic_with_uncertainty(face_rgb, pca, hyperplanes, num_samples=50):
    # baseline embedding / hashes
    base_emb = get_embedding(face_rgb)
    base_hv  = encode_embedding_to_hv(base_emb)
    base_nh  = compute_hash(pca.transform([base_emb])[0], hyperplanes)

    # enable MC-Dropout on resnet ...
    # collect multiple embeddings, compute NH & HV each time,
    # track Hamming distances to baseline,
    # compute std / var for NH and HV separately.
    ...
    return base_nh, base_hv, nh_std, hv_std
