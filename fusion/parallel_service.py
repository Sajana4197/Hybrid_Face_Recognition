# fusion/parallel_service.py
from pathlib import Path
import tempfile, os, cv2, numpy as np
import json
import time  # ✅ ADD: For performance timing

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
    ✅ NOW WITH DETAILED DISTANCE LOGGING
    """
    
    # ✅ START TIMING
    start_time = time.time()
    print("\n" + "="*80)
    print(f"🔍 STARTING FACE RECOGNITION MATCH")
    print("="*80)

    # 1️⃣ Face alignment
    align_start = time.time()
    face_rgb = _align_from_frame(frame_bgr)
    align_time = time.time() - align_start
    
    if face_rgb is None:
        print("❌ NO FACE DETECTED")
        print(f"⏱️  Alignment time: {align_time*1000:.1f}ms")
        print("="*80 + "\n")
        return {"decision": "NO_FACE", "person_id": None, "scores": {}}
    
    print(f"✅ Face aligned successfully: {face_rgb.shape}")
    print(f"⏱️  Alignment time: {align_time*1000:.1f}ms")

    # 2️⃣ Feature extraction (real modules)
    try:
        feature_start = time.time()
        
        print("\n📊 EXTRACTING FEATURES...")
        nh_start = time.time()
        probe_nh = compute_hash_bits(face_rgb)
        nh_time = time.time() - nh_start
        print(f"  ✓ NeuralHash computed: 96 bits | Time: {nh_time*1000:.1f}ms")

        emb_start = time.time()
        embedding = generate_embedding2(face_rgb)  # 512-D FaceNet features
        emb_time = time.time() - emb_start
        print(f"  ✓ FaceNet embedding: {embedding.shape} | Time: {emb_time*1000:.1f}ms")

        hdic_start = time.time()
        probe_hv = encode_embedding_to_hv(embedding).astype(np.float32)  # 10k-D HDIC HV
        hdic_time = time.time() - hdic_start
        print(f"  ✓ HDIC hypervector: {probe_hv.shape} | Time: {hdic_time*1000:.1f}ms")
        
        feature_time = time.time() - feature_start
        print(f"⏱️  Total feature extraction: {feature_time*1000:.1f}ms")

    except Exception as e:
        print(f"\n❌ ERROR during feature extraction: {e}")
        print("="*80 + "\n")
        return {
            "decision": "ERROR",
            "error": f"Encoding failed: {e}",
            "person_id": None,
            "scores": {},
        }

    # 3️⃣ Load both watchlists
    watchlist_start = time.time()
    nh_map, hd_map = _load_watchlists_jsonl(REPO_ROOT)
    person_ids = sorted(set(nh_map.keys()) & set(hd_map.keys()))
    watchlist_time = time.time() - watchlist_start
    
    if not person_ids:
        print("\n❌ ERROR: Empty watchlist")
        print("="*80 + "\n")
        return {"decision": "ERROR", "error": "Empty watchlist", "person_id": None, "scores": {}}
    
    print(f"\n📋 Loaded {len(person_ids)} persons from watchlist | Time: {watchlist_time*1000:.1f}ms")

    # 4️⃣ Score each person
    print("\n" + "-"*80)
    print("🎯 COMPUTING DISTANCES FOR ALL PERSONS")
    print("-"*80)
    print(f"{'Person ID':<15} {'NH Dist':<10} {'HDIC Dist':<12} {'S_NH':<8} {'S_HDIC':<10} {'S_final':<10} {'Status':<15}")
    print("-"*80)
    
    matching_start = time.time()
    best_pid, best_sfinal, best_metrics = None, -1.0, None
    all_scores = []
    
    for pid in person_ids:
        d_nh = _nh_min_distance(probe_nh, nh_map.get(pid, []))
        d_hdic = _hdic_min_distance(probe_hv, hd_map.get(pid, []))
        Snh = 1.0 - (d_nh / 96.0)
        Shdic_norm = 1.0 - (d_hdic / 10000.0)  # normalization for fusion
        Sfinal = (w_nh * Snh) + (w_hdic * Shdic_norm)
        
        # ✅ Determine pass/fail for each gate
        nh_pass = "✓" if d_nh < Tnh else "✗"
        hdic_pass = "✓" if d_hdic < Thdic else "✗"
        fused_pass = "✓" if Sfinal >= fused_th else "✗"
        status = f"NH{nh_pass} HD{hdic_pass} F{fused_pass}"
        
        # ✅ Print each person's scores
        print(f"{pid:<15} {d_nh:<10} {d_hdic:<12.0f} {Snh:<8.3f} {Shdic_norm:<10.3f} {Sfinal:<10.3f} {status:<15}")
        
        all_scores.append({
            "person_id": pid,
            "d_nh": d_nh,
            "d_hdic": d_hdic,
            "Snh": Snh,
            "Shdic_norm": Shdic_norm,
            "Sfinal": Sfinal
        })

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

    matching_time = time.time() - matching_start
    print("-"*80)
    print(f"⏱️  Matching time: {matching_time*1000:.1f}ms")

    if best_pid is None:
        print("\n❌ NO MATCH FOUND")
        print("="*80 + "\n")
        return {"decision": "NO_MATCH", "person_id": None, "scores": {}}

    # 5️⃣ Apply your original triple-gate rule
    is_match = (
        (best_metrics["d_nh"] < Tnh)
        and (best_metrics["d_hdic"] < Thdic)
        and (best_metrics["Sfinal"] >= fused_th)
    )
    decision = "MATCH" if is_match else "NO_MATCH"
    
    # ✅ FINAL RESULTS
    print("\n" + "="*80)
    print("🏆 BEST MATCH RESULTS")
    print("="*80)
    print(f"Person ID:         {best_pid}")
    print(f"Decision:          {decision}")
    print(f"\n📏 Distances:")
    print(f"  NH Distance:     {best_metrics['d_nh']} / 96  (threshold: {Tnh})")
    print(f"  HDIC Distance:   {best_metrics['d_hdic']:.0f} / 10000  (threshold: {Thdic})")
    print(f"\n📊 Similarity Scores:")
    print(f"  S_NH:            {best_metrics['Snh']:.4f}")
    print(f"  S_HDIC:          {best_metrics['Shdic_norm']:.4f}")
    print(f"  S_final:         {best_metrics['Sfinal']:.4f}  (threshold: {fused_th})")
    print(f"\n✅ Gate Status:")
    print(f"  NH Gate:         {'PASS ✓' if best_metrics['d_nh'] < Tnh else 'FAIL ✗'}")
    print(f"  HDIC Gate:       {'PASS ✓' if best_metrics['d_hdic'] < Thdic else 'FAIL ✗'}")
    print(f"  Fused Gate:      {'PASS ✓' if best_metrics['Sfinal'] >= fused_th else 'FAIL ✗'}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  TOTAL TIME: {total_time*1000:.1f}ms")
    print("  ├─ Alignment:    {:.1f}ms ({:.1f}%)".format(align_time*1000, align_time/total_time*100))
    print("  ├─ Features:     {:.1f}ms ({:.1f}%)".format(feature_time*1000, feature_time/total_time*100))
    print("  ├─ Watchlist:    {:.1f}ms ({:.1f}%)".format(watchlist_time*1000, watchlist_time/total_time*100))
    print("  └─ Matching:     {:.1f}ms ({:.1f}%)".format(matching_time*1000, matching_time/total_time*100))
    print("="*80 + "\n")

    return {"decision": decision, "person_id": best_pid, "scores": best_metrics}