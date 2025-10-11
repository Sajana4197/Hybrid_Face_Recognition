from pathlib import Path
import json
import numpy as np
from typing import Dict, List

def load_watchlists(repo_root: Path) -> Dict[str, dict]:
    """
    Merge db/watchlist_neuralhash.jsonl and db/watchlist_hdic.jsonl by person_id.
    Keeps only persons that have BOTH modalities.
    Returns:
      {
        person_id: {
          "name": str,
          "nh_hashes": [np.uint8(96), ...],
          "hdic_prototypes": [np.float32(10000), ...]
        }, ...
      }
    """
    db_dir = repo_root / "db"
    nh_file = db_dir / "watchlist_neuralhash.jsonl"
    hdic_file = db_dir / "watchlist_hdic.jsonl"

    people = {}

    # NH watchlist
    if nh_file.exists():
        with nh_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                if not pid:
                    continue
                name = rec.get("name", pid)
                hashes = rec.get("hashes", [])
                nh_arrays: List[np.ndarray] = []
                for h in hashes:
                    a = np.asarray(h, dtype=np.uint8).reshape(-1)
                    if a.size == 96:  # 96-bit NH
                        nh_arrays.append(a)
                entry = people.setdefault(pid, {"name": name, "nh_hashes": [], "hdic_prototypes": []})
                entry["name"] = name
                entry["nh_hashes"].extend(nh_arrays)

    # HDIC watchlist
    if hdic_file.exists():
        with hdic_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = rec.get("person_id") or rec.get("id") or rec.get("pid")
                if not pid:
                    continue
                name = rec.get("name", pid)
                prototypes = rec.get("prototypes", {})
                hv_list: List[np.ndarray] = []
                # keep cluster_0, cluster_1, cluster_2 in order if present
                for key in sorted(prototypes.keys()):
                    v = prototypes[key]
                    a = np.asarray(v, dtype=np.float32).reshape(-1)
                    if a.size > 0:
                        hv_list.append(a)
                entry = people.setdefault(pid, {"name": name, "nh_hashes": [], "hdic_prototypes": []})
                entry["name"] = name
                entry["hdic_prototypes"].extend(hv_list)

    # Require both modalities
    merged = {pid: rec for pid, rec in people.items() if rec["nh_hashes"] and rec["hdic_prototypes"]}
    return merged
