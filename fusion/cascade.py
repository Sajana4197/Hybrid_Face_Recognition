import numpy as np
from common.hamming import hamming_distance_bits

def shortlist_by_neuralhash(query_bits, nh_db, T_NH_reject, K):
    per_person_best = []
    for rec in nh_db:
        best = 1e9
        for hv in rec.get("hashes", []):
            bits = np.array(hv, dtype=np.uint8)  # stored as list of 0/1
            d = hamming_distance_bits(query_bits, bits)
            if d < best:
                best = d
        per_person_best.append((rec["person_id"], best))
    per_person_best.sort(key=lambda x: x[1])
    if not per_person_best:
        return [], True
    if per_person_best[0][1] > T_NH_reject:
        return per_person_best[:K], True
    return per_person_best[:K], False

def hdic_confirm(query_hv_bits, person_id, hdic_db):
    recs = [r for r in hdic_db if r["person_id"] == person_id]
    if not recs:
        return 1e9, "none"
    rec = recs[0]
    best = 1e9
    best_cluster = "none"
    for cname, proto in rec.get("prototypes", {}).items():
        proto_bits = np.array(proto, dtype=np.uint8)  # stored as list of 0/1
        d = hamming_distance_bits(query_hv_bits, proto_bits)
        if d < best:
            best = d
            best_cluster = cname
    return best, best_cluster
