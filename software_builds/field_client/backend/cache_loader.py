"""
Fast watchlist loader using NPZ cache.  
Falls back to JSONL if cache doesn't exist.
"""
from pathlib import Path
import numpy as np
from typing import Dict, Tuple
import json


def load_from_npz_cache(repo_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np. ndarray], Dict[str, str]]:
    """
    Load watchlist from NPZ cache for fast startup.
    
    Returns:
        nh_map: {person_id: np.ndarray (N_hashes, 96) uint8}
        hdic_map: {person_id: np.ndarray (N_clusters, 10000) uint8/float32}
        name_map: {person_id: name}
    """
    cache_file = repo_root / "db" / "watchlist_cache.npz"
    
    if not cache_file.exists():
        print("[WARN] NPZ cache not found, falling back to JSONL...")
        return load_from_jsonl_fallback(repo_root)
    
    try:
        print(f"[INFO] Loading watchlist from NPZ cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=False)
        
        person_ids = data['person_ids']. tolist()
        person_names = data. get('person_names', np.array([])).tolist()
        
        nh_map = {}
        hdic_map = {}
        name_map = {}
        
        for i, pid in enumerate(person_ids):
            # Load NeuralHash data
            nh_key = f'nh_{i}'
            if nh_key in data:
                nh_map[pid] = data[nh_key]
            
            # Load HDIC data
            hdic_key = f'hdic_{i}'
            if hdic_key in data:
                hdic_map[pid] = data[hdic_key]
            
            # Load name
            if i < len(person_names):
                name_map[pid] = person_names[i]
            else:
                name_map[pid] = pid
        
        print(f"[INFO] ✅ Loaded {len(person_ids)} persons from NPZ cache")
        print(f"[INFO]    NH entries: {len(nh_map)}, HDIC entries: {len(hdic_map)}")
        
        return nh_map, hdic_map, name_map
        
    except Exception as e:
        print(f"[ERROR] Failed to load NPZ cache: {e}")
        print("[INFO] Falling back to JSONL...")
        return load_from_jsonl_fallback(repo_root)


def load_from_jsonl_fallback(repo_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Fallback loader from JSONL files (slower but always works).
    """
    db_dir = repo_root / "db"
    nh_file = db_dir / "watchlist_neuralhash.jsonl"
    hdic_file = db_dir / "watchlist_hdic.jsonl"
    
    nh_map = {}
    hdic_map = {}
    name_map = {}
    
    print("[INFO] Loading from JSONL files...")
    
    # Load NeuralHash
    if nh_file.exists():
        with open(nh_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec. get("person_id") or rec.get("id")
                    if not pid:
                        continue
                    
                    name_map[pid] = rec.get("name", pid)
                    hashes = rec.get("hashes", [])
                    if hashes:
                        nh_map[pid] = np. array(hashes, dtype=np.uint8)
                except Exception as e:
                    print(f"[WARN] Skipped invalid NH record: {e}")
    
    # Load HDIC
    if hdic_file.exists():
        with open(hdic_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec.get("person_id") or rec. get("id")
                    if not pid:
                        continue
                    
                    prototypes = rec.get("prototypes", {})
                    if prototypes:
                        proto_list = [np.array(v, dtype=np.uint8) for v in prototypes.values()]
                        if proto_list:
                            hdic_map[pid] = np.vstack(proto_list)
                except Exception as e:
                    print(f"[WARN] Skipped invalid HDIC record: {e}")
    
    print(f"[INFO] ✅ Loaded from JSONL: NH={len(nh_map)}, HDIC={len(hdic_map)}")
    
    return nh_map, hdic_map, name_map


def check_cache_freshness(repo_root: Path) -> bool:
    """
    Check if NPZ cache is up-to-date compared to JSONL files.
    Returns True if cache is fresh, False if it needs rebuild.
    """
    db_dir = repo_root / "db"
    cache_file = db_dir / "watchlist_cache.npz"
    nh_file = db_dir / "watchlist_neuralhash.jsonl"
    hdic_file = db_dir / "watchlist_hdic.jsonl"
    
    if not cache_file.exists():
        return False
    
    cache_mtime = cache_file.stat(). st_mtime
    
    # Check if JSONL files are newer
    if nh_file.exists() and nh_file.stat().st_mtime > cache_mtime:
        return False
    
    if hdic_file.exists() and hdic_file.stat().st_mtime > cache_mtime:
        return False
    
    return True