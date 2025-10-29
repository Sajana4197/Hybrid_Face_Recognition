import os, json, shutil
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
BASE = Path(__file__).parent / "verifications"
PENDING = BASE / "pending"
DECIDED = BASE / "decided"
LEDGER = BASE / "verifications.jsonl"
for d in (PENDING, DECIDED):
    d.mkdir(parents=True, exist_ok=True)
if not LEDGER.exists():
    LEDGER.write_text("")

def _append(row: dict):
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def _read_all():
    if not LEDGER.exists(): return []
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]

def _save_all(rows: list):
    """Save all rows back to ledger"""
    with open(f"{LEDGER}.tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(f"{LEDGER}.tmp", LEDGER)

def enqueue_case(person_id: str, score: float, img_bytes: bytes):
    """Store a pending verification case locally"""
    # Use the same timestamp format as admin alerts
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"{person_id}_{ts}.jpg"
    fpath = PENDING / fname
    with open(fpath, "wb") as f:
        f.write(img_bytes)
    row = {
        "person_id": person_id,
        "timestamp": ts,  # ✅ This now matches admin timestamp format
        "status": "pending",
        "score": round(score, 3),
        "image": fname,
        "decision_time": None,
    }
    _append(row)
    return row

@router.get("/verifications")
def list_verifications():
    """List all verification cases"""
    rows = _read_all()
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return {"items": rows}

@router.post("/verifications/update")
def update_verification(data: dict):
    """
    Called by Admin when a decision is made.
    Expected data: {person_id, timestamp, status: "confirmed"|"rejected"}
    """
    items = _read_all()
    found = False
    
    for r in items:
        # ✅ Match by person_id and timestamp
        if r["person_id"] == data["person_id"] and r["timestamp"] == data["timestamp"]:
            r["status"] = data["status"]
            r["decision_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            
            # Move image from pending to decided
            src = PENDING / r["image"]
            if src.exists():
                shutil.move(src, DECIDED / r["image"])
            break
    
    if not found:
        raise HTTPException(404, "Verification not found")

    _save_all(items)
    return {"ok": True, "status": "updated"}

@router.get("/verifications/image/{fname}")
def get_image(fname: str):
    """Serve verification images"""
    for d in (PENDING, DECIDED):
        f = d / fname
        if f.exists():
            return FileResponse(f)
    raise HTTPException(404, "Image not found")