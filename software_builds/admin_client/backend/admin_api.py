# software_builds/admin_client/backend/admin_api.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import numpy as np, os, json
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime

# ---- Robust imports for your shared core ----
try:
    from software_builds.common_core.packed_store import PackedStore
    from software_builds.common_core.cache import rebuild_cache_async
    from software_builds.common_core.settings import NH_DIR, HD_DIR
except Exception:
    # Fallbacks if running without package prefix
    from db.packed_store import PackedStore  # adjust if your path differs
    from db.cache_manager import rebuild_cache_async
    # If you don’t have a central settings file, point to your real folders:
    from pathlib import Path as _P
    REPO_ROOT = _P(__file__).resolve().parents[2]
    NH_DIR = REPO_ROOT / "db" / "nh_packed"
    HD_DIR = REPO_ROOT / "db" / "hdic_packed"

# Face pipeline
from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from hdic.feature_extractor import generate_embedding2
from hdic.encode_hv import encode_embedding_to_hv
from hdic.cluster_enroll import build_cluster_prototypes

router = APIRouter()

# ---------- Packed DB listing ----------
def _list_packed():
    nh = PackedStore(NH_DIR, bits=96)
    hd = PackedStore(HD_DIR, bits=10000)

    NH_ROWS, NH_OFFS, NH_PIDS = nh.load_memmap()
    HD_ROWS, HD_OFFS, HD_PIDS = hd.load_memmap()

    nh_names_p = NH_DIR / "person_names.json"
    hd_names_p = HD_DIR / "person_names.json"
    NH_NAMES = json.loads(nh_names_p.read_text(encoding="utf-8")) if nh_names_p.exists() else []
    HD_NAMES = json.loads(hd_names_p.read_text(encoding="utf-8")) if hd_names_p.exists() else []

    info = {}
    for i, pid in enumerate(NH_PIDS):
        info[pid] = dict(
            name=NH_NAMES[i] if i < len(NH_NAMES) else "",
            nh_count=int(NH_OFFS[i + 1] - NH_OFFS[i]),
            hdic_count=0,
        )
    for i, pid in enumerate(HD_PIDS):
        cnt = int(HD_OFFS[i + 1] - HD_OFFS[i])
        if pid in info:
            info[pid]["hdic_count"] = cnt
            if not info[pid]["name"] and i < len(HD_NAMES):
                info[pid]["name"] = HD_NAMES[i]
        else:
            info[pid] = dict(
                name=HD_NAMES[i] if i < len(HD_NAMES) else "",
                nh_count=0,
                hdic_count=cnt,
            )
    return [dict(person_id=k, **v) for k, v in sorted(info.items())]

@router.get("/list")
def list_people():
    return {"persons": _list_packed()}

# ---------- Enroll / Delete ----------
@router.post("/enroll")
async def enroll(
    person_id: str = Form(...),
    name: str = Form(""),
    files: List[UploadFile] = File(...),
    overwrite: bool = Form(False),
):
    nh = PackedStore(NH_DIR, bits=96)
    hd = PackedStore(HD_DIR, bits=10000)
    _, _, pids = nh.load_memmap()

    if person_id in pids and not overwrite:
        raise HTTPException(409, f"Person '{person_id}' exists (set overwrite=true to replace)")
    if person_id in pids and overwrite:
        _rewrite_without_pid(NH_DIR, person_id)
        _rewrite_without_pid(HD_DIR, person_id)

    nh_rows, hv_rows = [], []
    added = failed = 0

    for uf in files:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await uf.read())
            tmp_path = tmp.name
        try:
            face = load_and_align(tmp_path, output_size=(160, 160), normalize=False)
            if face is None:
                failed += 1
                continue

            nh_bits = compute_hash_bits(face).astype(np.uint8).ravel()
            emb = generate_embedding2(face)
            hv_bits = encode_embedding_to_hv(emb).astype(np.uint8).ravel()

            nh_rows.append(nh_bits)
            hv_rows.append(hv_bits)
            added += 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if added == 0:
        raise HTTPException(400, "No valid faces")

    # Prototypes
    try:
        protos = build_cluster_prototypes(hv_rows, num_clusters=3)
        if isinstance(protos, dict):
            protos_mat = np.stack([np.asarray(v) for v in protos.values()], axis=0)
        else:
            protos_mat = np.asarray(protos)
        protos_mat = protos_mat.astype(np.uint8, copy=False)
        if protos_mat.max() > 1:
            protos_mat = (protos_mat >= 0.5).astype(np.uint8)
        if protos_mat.shape[1] != 10000:
            raise ValueError("Prototypes must be (K, 10000)")
    except Exception:
        # fallback: 1 prototype by rounding mean
        protos_mat = np.asarray([np.round(np.mean(hv_rows, axis=0))], dtype=np.uint8)

    nh_rows = np.stack(nh_rows, axis=0).astype(np.uint8, copy=False)
    nh.append_person(person_id, name, nh_rows)
    hd.append_person(person_id, name, protos_mat)

    rebuild_cache_async()
    return {
        "status": "ok",
        "person_id": person_id,
        "added_images": int(added),
        "nh_rows": int(nh_rows.shape[0]),
        "hdic_clusters": int(protos_mat.shape[0]),
        "cache_status": "rebuilding",
    }

@router.delete("/delete/{person_id}")
def delete_person(person_id: str):
    found1 = _rewrite_without_pid(NH_DIR, person_id)
    found2 = _rewrite_without_pid(HD_DIR, person_id)
    if not (found1 or found2):
        raise HTTPException(404, f"Person '{person_id}' not found")
    rebuild_cache_async()
    return {"status": "deleted", "person_id": person_id, "cache_status": "rebuilding"}

# ---------- Manual checks (alerts) ----------
# Public static mount is in main.py -> /uploads
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

ALERTS = Path(__file__).resolve().parent / "manual_checks.jsonl"

def _load_alerts():
    rows = []
    if ALERTS.exists():
        for line in ALERTS.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def _save_alerts(rows):
    tmp = ALERTS.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    os.replace(tmp, ALERTS)

@router.post("/manual_check/receive")
async def receive_manual_check(
    file: UploadFile = File(...),
    person_id: str = Form(...),
    score: float = Form(...),
    timestamp: str = Form(...),
):
    filename = f"{person_id}_{timestamp}.jpg".replace(":", "-")
    out = UPLOADS_DIR / filename
    out.write_bytes(await file.read())

    rows = _load_alerts()
    rows.append({
        "person_id": person_id,
        "score": float(score),
        "similarity": float(score),             # <-- alias for UI that used 'similarity'
        "timestamp": timestamp,
        "file": str(filename),
        "file_path": f"/uploads/{filename}",    # <-- what the UI loads
        "status": "pending",
    })
    _save_alerts(rows)
    return {"ok": True}

@router.get("/manual_check/list")
def list_alerts():
    return {"alerts": _load_alerts()}

@router.post("/manual_check/decision")
def update_decision(
    person_id: str = Form(...),
    timestamp: str = Form(...),
    decision: str = Form(...),
):
    decision = decision.strip().lower()
    if decision not in ("confirm", "reject"):
        raise HTTPException(400, "decision must be confirm|reject")
    rows = _load_alerts()
    found = False
    for r in rows:
        if r.get("person_id") == person_id and r.get("timestamp") == timestamp:
            r["status"] = "confirmed" if decision == "confirm" else "rejected"
            r["decision_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if not found:
        raise HTTPException(404, "alert not found")
    _save_alerts(rows)
    return {"ok": True}

# ---------- helpers ----------
def _atomic_save_npy(path: Path, arr: np.ndarray):
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)

def _atomic_save_json(path: Path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)

def _rewrite_without_pid(base_dir: Path, pid: str):
    data_p, offs_p = base_dir / "data.npy", base_dir / "offsets.npy"
    pids_p, names_p, meta_p = base_dir / "person_ids.json", base_dir / "person_names.json", base_dir / "meta.json"
    if not (data_p.exists() and offs_p.exists() and pids_p.exists() and meta_p.exists()):
        return False
    data = np.load(data_p, allow_pickle=False)
    offs = np.load(offs_p, allow_pickle=False)
    pids = json.loads(pids_p.read_text(encoding="utf-8"))
    names = json.loads(names_p.read_text(encoding="utf-8")) if names_p.exists() else ["" for _ in pids]
    if pid not in pids:
        return False
    idx = pids.index(pid)

    kept_blocks = []
    for i in range(len(pids)):
        if i == idx:
            continue
        s, e = int(offs[i]), int(offs[i + 1])
        kept_blocks.append(data[s:e])
    new_data = np.concatenate(kept_blocks, axis=0) if kept_blocks else np.zeros((0, data.shape[1]), dtype=np.uint64)

    new_offs = [0]
    for i in range(len(pids)):
        if i == idx:
            continue
        s, e = int(offs[i]), int(offs[i + 1])
        new_offs.append(new_offs[-1] + (e - s))
    new_offs = np.asarray(new_offs, dtype=np.int64)

    new_pids = [p for i, p in enumerate(pids) if i != idx]
    new_names = [n for i, n in enumerate(names) if i != idx]

    _atomic_save_npy(data_p, new_data)
    _atomic_save_npy(offs_p, new_offs)
    _atomic_save_json(pids_p, new_pids)
    _atomic_save_json(names_p, new_names)
    return True
