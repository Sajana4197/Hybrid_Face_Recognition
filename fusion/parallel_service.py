# fusion/parallel_service.py
from pathlib import Path
import cv2, numpy as np
import time
import sys
import atexit

# ✅ FIX: Import from absolute path, not relative
REPO_ROOT = Path(__file__).resolve().parents[1]

# Add the db directory to Python path with higher priority
sys.path.insert(0, str(REPO_ROOT / "db"))

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from hdic.encode_hv import encode_embedding_to_hv
from hdic.feature_extractor import generate_embedding2

# ✅ NOW import cache manager (will find the correct one)
from db.cache_manager import ensure_cache_exists, start_cache_watcher, cache_exists

# ---------------------------------------------------------
# CONFIGURATION & SMART CACHE LOADING
# ---------------------------------------------------------
CACHE_FILE = REPO_ROOT / "db" / "watchlist_cache.npz"

# Global cache state
NH_MAP = {}
HD_MAP = {}
PERSON_NAMES = {}
_cache_loaded = False
_cache_watcher = None


def _load_watchlist_cache():
    """Load pre-built NumPy cache with error handling"""
    global _cache_loaded
    
    if not CACHE_FILE.exists():
        print(f"[WARN] Cache file not found: {CACHE_FILE}")
        return {}, {}, {}
    
    try:
        cache = np.load(CACHE_FILE, allow_pickle=False)
        person_ids = cache.get('person_ids', np.array([])).tolist()
        person_names = cache.get('person_names', np.array([])).tolist()
        
        if len(person_ids) == 0:
            print("[INFO] Cache loaded but contains 0 persons (empty watchlist)")
            return {}, {}, {}
        
        nh_map = {}
        hdic_map = {}
        
        for i, pid in enumerate(person_ids):
            try:
                nh_map[pid] = cache[f'nh_{i}']
                hdic_map[pid] = cache[f'hdic_{i}']
            except KeyError as e:
                print(f"[WARN] Skipping person {pid}: missing data ({e})")
                continue
        
        _cache_loaded = True
        return nh_map, hdic_map, dict(zip(person_ids, person_names))
        
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        return {}, {}, {}


def reload_cache():
    """
    Reload cache from disk (called when watchlist changes).
    This allows the application to pick up new enrollments without restart.
    """
    global NH_MAP, HD_MAP, PERSON_NAMES, _cache_loaded
    
    print("[INFO] 🔄 Reloading cache...")
    
    try:
        nh_new, hd_new, names_new = _load_watchlist_cache()
        
        # Only update if load was successful
        if nh_new or hd_new:
            NH_MAP = nh_new
            HD_MAP = hd_new
            PERSON_NAMES = names_new
            print(f"[INFO] ✅ Cache reloaded: {len(NH_MAP)} persons")
        else:
            print("[WARN] Cache reload returned empty data")
            
    except Exception as e:
        print(f"[ERROR] Cache reload failed: {e}")
        print("[INFO] Keeping existing cache in memory")


def initialize_cache():
    """
    Initialize cache system on module load.
    This is called automatically when the module is imported.
    """
    global NH_MAP, HD_MAP, PERSON_NAMES, _cache_watcher
    
    print("\n" + "="*70)
    print("🚀 INITIALIZING HYBRID FACE RECOGNITION SYSTEM")
    print("="*70)
    
    # Step 1: Ensure cache exists
    print("[1/3] Checking watchlist cache...")
    
    try:
        cache_ready = ensure_cache_exists()
        if not cache_ready:
            print("[WARN] Cache initialization had issues")
    except Exception as e:
        print(f"[ERROR] Cache check failed: {e}")
        print("[INFO] Attempting to create empty cache...")
        
        # Create minimal empty cache as fallback
        try:
            from db.build_cache import build_cache
            build_cache(silent=False)
        except Exception as build_error:
            print(f"[ERROR] Could not build cache: {build_error}")
            print("[WARN] Application will start with empty watchlist")
    
    # Step 2: Load cache into memory
    print("\n[2/3] Loading watchlist into memory...")
    cache_start = time.time()
    
    try:
        NH_MAP, HD_MAP, PERSON_NAMES = _load_watchlist_cache()
        cache_time = time.time() - cache_start
        
        if len(NH_MAP) > 0:
            print(f"[INFO] ✅ Loaded {len(NH_MAP)} persons in {cache_time*1000:.1f}ms")
            
            # Show cache file info
            if CACHE_FILE.exists():
                cache_size_kb = CACHE_FILE.stat().st_size / 1024
                cache_age_sec = time.time() - CACHE_FILE.stat().st_mtime
                print(f"[INFO] Cache size: {cache_size_kb:.1f} KB")
                print(f"[INFO] Cache age: {cache_age_sec/60:.1f} minutes")
        else:
            print("[INFO] ⚠️  Watchlist is empty (no persons enrolled)")
            print("[INFO] Application ready but will return NO_MATCH until enrollment")
            
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        print("[WARN] Starting with empty watchlist")
        NH_MAP, HD_MAP, PERSON_NAMES = {}, {}, {}
    
    # Step 3: Start automatic cache watcher
    print("\n[3/3] Starting automatic cache watcher...")
    
    try:
        _cache_watcher = start_cache_watcher(check_interval=30)
        print("[INFO] ✅ Cache watcher active (checks every 30s)")
        
        # Register shutdown handler
        def _shutdown():
            if _cache_watcher:
                print("\n[INFO] Stopping cache watcher...")
                _cache_watcher.stop()
        
        atexit.register(_shutdown)
        
    except Exception as e:
        print(f"[WARN] Cache watcher failed to start: {e}")
        print("[INFO] Cache won't auto-reload (manual restart needed for updates)")
    
    print("\n" + "="*70)
    print(f"✅ SYSTEM READY | {len(NH_MAP)} persons loaded")
    print("="*70 + "\n")


# ✅ AUTOMATIC INITIALIZATION
# This runs when the module is imported
initialize_cache()


# ---------------------------------------------------------
# OPTIMIZED DISTANCE FUNCTIONS
# ---------------------------------------------------------

def _nh_min_distance_vectorized(probe_bits: np.ndarray, hashes_matrix: np.ndarray) -> int:
    """Vectorized Hamming distance for NeuralHash"""
    if hashes_matrix.size == 0:
        return 96
    distances = np.sum(hashes_matrix != probe_bits, axis=1)
    return int(np.min(distances))


def _hdic_min_distance_vectorized(probe_hv: np.ndarray, prototypes_matrix: np.ndarray) -> float:
    """Vectorized Hamming distance for HDIC"""
    if prototypes_matrix.size == 0:
        return 1e9
    distances = np.sum(prototypes_matrix != probe_hv, axis=1)
    return float(np.min(distances))


def _align_from_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Direct alignment from BGR frame"""
    import tempfile, os
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, frame_bgr)
    try:
        face_rgb = load_and_align(tmp_path, output_size=(160, 160), normalize=False)
        return face_rgb
    finally:
        try: os.remove(tmp_path)
        except: pass


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
    ✅ FULLY AUTOMATIC: Uses cached data that auto-updates
    ✅ PRODUCTION SAFE: Handles empty watchlist gracefully
    """
    
    start_time = time.time()
    print("\n" + "="*80)
    print(f"🔍 STARTING FACE RECOGNITION MATCH")
    print("="*80)

    # ✅ Check if watchlist is loaded
    if len(NH_MAP) == 0 or len(HD_MAP) == 0:
        print("⚠️  WARNING: Watchlist is empty (no persons enrolled)")
        print("="*80 + "\n")
        return {
            "decision": "NO_WATCHLIST",
            "person_id": None,
            "scores": {},
            "message": "No persons enrolled in watchlist"
        }

    # 1️⃣ Face alignment
    align_start = time.time()
    face_rgb = _align_from_frame(frame_bgr)
    align_time = time.time() - align_start
    
    if face_rgb is None:
        print("❌ NO FACE DETECTED")
        print(f"⏱️  Alignment: {align_time*1000:.1f}ms")
        print("="*80 + "\n")
        return {"decision": "NO_FACE", "person_id": None, "scores": {}}
    
    print(f"✅ Face aligned: {face_rgb.shape} | {align_time*1000:.1f}ms")

    # 2️⃣ Feature extraction
    try:
        feature_start = time.time()
        
        probe_nh = compute_hash_bits(face_rgb)
        embedding = generate_embedding2(face_rgb)
        probe_hv = encode_embedding_to_hv(embedding).astype(np.float32)
        
        feature_time = time.time() - feature_start
        print(f"✅ Features extracted | {feature_time*1000:.1f}ms")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("="*80 + "\n")
        return {"decision": "ERROR", "error": str(e), "person_id": None, "scores": {}}

    # 3️⃣ Use cached watchlist
    person_ids = list(NH_MAP.keys())
    print(f"📋 Matching against {len(person_ids)} persons")

    # 4️⃣ Vectorized scoring
    print("-"*80)
    
    matching_start = time.time()
    best_pid, best_sfinal, best_metrics = None, -1.0, None
    
    for pid in person_ids:
        d_nh = _nh_min_distance_vectorized(probe_nh, NH_MAP[pid])
        d_hdic = _hdic_min_distance_vectorized(probe_hv, HD_MAP[pid])
        
        Snh = 1.0 - (d_nh / 96.0)
        Shdic_norm = 1.0 - (d_hdic / 10000.0)
        Sfinal = (w_nh * Snh) + (w_hdic * Shdic_norm)
        
        name = PERSON_NAMES.get(pid, pid)[:18]
        print(f"{name:<20} NH:{d_nh:<3} HDIC:{d_hdic:<5.0f} Final:{Sfinal:.3f}")

        if Sfinal > best_sfinal:
            best_sfinal = Sfinal
            best_pid = pid
            best_metrics = {
                "d_nh": d_nh,
                "d_hdic": d_hdic,
                "Snh": Snh,
                "Shdic_norm": Shdic_norm,
                "Sfinal": Sfinal
            }
            
            # Early exit
            if (d_nh < Tnh and d_hdic < Thdic and Sfinal >= 0.95):
                print(f"⚡ Early exit: High confidence!")
                break

    matching_time = time.time() - matching_start
    print(f"-"*80)
    print(f"⏱️  Matching: {matching_time*1000:.1f}ms")

    if best_pid is None:
        print("\n❌ NO MATCH\n" + "="*80 + "\n")
        return {"decision": "NO_MATCH", "person_id": None, "scores": {}}

    # 5️⃣ Final decision
    is_match = (
        best_metrics["d_nh"] < Tnh 
        and best_metrics["d_hdic"] < Thdic 
        and best_metrics["Sfinal"] >= fused_th
    )
    decision = "MATCH" if is_match else "NO_MATCH"
    
    total_time = time.time() - start_time
    print(f"\n🏆 {decision}: {PERSON_NAMES.get(best_pid, best_pid)}")
    print(f"⏱️  TOTAL: {total_time*1000:.1f}ms")
    print("="*80 + "\n")

    return {"decision": decision, "person_id": best_pid, "scores": best_metrics}