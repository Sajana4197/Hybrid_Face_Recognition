"""
Automatic watchlist cache builder for optimized performance.
Converts JSONL watchlists to binary NumPy format.
"""
import json
import numpy as np
from pathlib import Path
import time

REPO_ROOT = Path(__file__).parent.parent
DB_DIR = REPO_ROOT / "db"


def build_cache(silent=False):
    """
    Convert JSONL watchlists to optimized NumPy binary format.
    
    Args:
        silent: If True, suppress print statements
        
    Returns:
        Path to created cache file
    """
    if not silent:
        print("🔨 Building watchlist cache...")
    
    start_time = time.time()
    
    # Load NeuralHash
    nh_file = DB_DIR / "watchlist_neuralhash.jsonl"
    nh_data = {}
    person_names = {}
    
    if nh_file.exists():
        with open(nh_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec.get("person_id") or rec.get("id")
                    if not pid:
                        continue
                    
                    person_names[pid] = rec.get("name", pid)
                    hashes = rec.get("hashes", [])
                    
                    # Convert to uint8 array: shape (N_hashes, 96)
                    if hashes:
                        nh_array = np.array(hashes, dtype=np.uint8)
                        nh_data[pid] = nh_array
                except Exception as e:
                    if not silent:
                        print(f"  ⚠️  Skipped invalid NH record: {e}")
                    
    if not silent:
        print(f"  ✓ Loaded {len(nh_data)} NeuralHash entries")
    
    # Load HDIC
    hdic_file = DB_DIR / "watchlist_hdic.jsonl"
    hdic_data = {}
    
    if hdic_file.exists():
        with open(hdic_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec.get("person_id") or rec.get("id")
                    if not pid:
                        continue
                    
                    prototypes = rec.get("prototypes", {})
                    if prototypes:
                        # Convert to float32 array: shape (N_clusters, 10000)
                        proto_list = [np.array(v, dtype=np.float32) for v in prototypes.values()]
                        if proto_list:
                            hdic_array = np.vstack(proto_list)
                            hdic_data[pid] = hdic_array
                except Exception as e:
                    if not silent:
                        print(f"  ⚠️  Skipped invalid HDIC record: {e}")
                        
    if not silent:
        print(f"  ✓ Loaded {len(hdic_data)} HDIC entries")
    
    # Find persons with both modalities
    valid_pids = sorted(set(nh_data.keys()) & set(hdic_data.keys()))
    
    if len(valid_pids) == 0:
        if not silent:
            print("  ⚠️  No valid persons found (need both NH and HDIC)")
        # Create empty cache
        cache_file = DB_DIR / "watchlist_cache.npz"
        np.savez_compressed(cache_file, person_ids=np.array([], dtype='U32'))
        return cache_file
    
    if not silent:
        print(f"  ✓ {len(valid_pids)} persons have both NH and HDIC")
    
    # Save as compressed NumPy archive
    cache_file = DB_DIR / "watchlist_cache.npz"
    
    save_dict = {
        'person_ids': np.array(valid_pids, dtype='U32'),  # Unicode strings
        'person_names': np.array([person_names.get(pid, pid) for pid in valid_pids], dtype='U64')
    }
    
    # Save NH data
    for i, pid in enumerate(valid_pids):
        save_dict[f'nh_{i}'] = nh_data[pid]
        
    # Save HDIC data
    for i, pid in enumerate(valid_pids):
        save_dict[f'hdic_{i}'] = hdic_data[pid]
    
    np.savez_compressed(cache_file, **save_dict)
    
    elapsed = time.time() - start_time
    cache_size_kb = cache_file.stat().st_size / 1024
    
    if not silent:
        print(f"  ✓ Saved cache to {cache_file.name}")
        print(f"  ✓ Cache size: {cache_size_kb:.1f} KB")
        print(f"  ✓ Build time: {elapsed*1000:.0f}ms")
        print("✅ Cache build complete!\n")
    
    return cache_file


if __name__ == "__main__":
    build_cache()