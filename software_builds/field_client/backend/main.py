import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import requests 
import json, yaml, uvicorn, cv2, numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from math import ceil
from software_builds.field_client.backend.verifications import router as ver_router, enqueue_case

# ---- Internal project imports ----
from fusion.parallel_service import match_frame
from software_builds.field_client.backend.loader import load_watchlists
from software_builds.field_client.backend.verifications import router as ver_router, enqueue_case

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# ----- FastAPI app -----
app = FastAPI(title="Hybrid Field Client (Parallel NH+HDIC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ver_router)

# ----- Config loading -----
CFG_PATH = Path(__file__).parent / "config.yaml"
CONFIG = yaml.safe_load(CFG_PATH.read_text())
Tnh     = float(CONFIG.get("Tnh", 30))
Thdic   = float(CONFIG.get("Thdic", 3100))
w_nh    = float(CONFIG.get("w_nh", 0.5))
w_hdic  = float(CONFIG.get("w_hdic", 0.5))
T_final = float(CONFIG.get("T_final", 0.7))
PORT    = int(CONFIG.get("port", 5001))
LOGPATH = Path(__file__).parent / (CONFIG.get("log_path", "logs/matches.jsonl"))
LOGPATH.parent.mkdir(parents=True, exist_ok=True)

# ✅ NEW: Load Admin API address from config.yaml
ADMIN_API = CONFIG.get("admin_api", "http://127.0.0.1:5002")

# Load merged watchlist (NH + HDIC)
REPO_ROOT = Path(__file__).resolve().parents[3]
PERSONS = {}

def reload_watchlist():
    """
    Reload watchlist from NPZ cache ONLY (super fast). 
    Does NOT rebuild cache - admin does that.
    """
    global PERSONS
    
    print("\n" + "="*60)
    print("[RELOAD] Fast reload from NPZ cache")
    print("="*60)
    
    # Reload fusion service data from NPZ cache (FAST!)
    from fusion import parallel_service
    count = parallel_service.load_watchlist_data()
    
    # Build PERSONS dict from the fusion engine data (fast, no file I/O)
    PERSONS = {}
    for pid in parallel_service. PERSON_IDS:
        name = parallel_service.NAME_MAP. get(pid, pid)
        PERSONS[pid] = {
            "name": name,
            "nh_hashes": parallel_service.NH_MAP.get(pid, []),
            "hdic_prototypes": parallel_service.HD_MAP.get(pid, [])
        }
    
    print(f"[INFO] ✅ Reload complete in <1 second!")
    print(f"[INFO]    Loaded {count} persons from NPZ cache")
    print("="*60 + "\n")
    
    return count

# Initial load
reload_watchlist()

# ---------- Utility ----------
def log_match(entry: dict):
    with LOGPATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def send_manual_alert(person_id: str, score: float, image_bytes: bytes, timestamp: str):
    """
    Send captured best-match frame to Admin for manual verification.
    Includes person ID, score, and timestamp.
    """
    try:
        files = {"file": ("match.jpg", BytesIO(image_bytes), "image/jpeg")}
        data = {"person_id": person_id, "score": score, "timestamp": timestamp}
        r = requests.post(
            f"{ADMIN_API}/manual_check/receive",
            files=files,
            data=data,
            timeout=8,
        )
        print(f"[INFO] Sent manual verification alert: {r.status_code}")
    except Exception as e:
        print("[WARN] Failed to send manual verification alert:", e)


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "watchlist_size": len(PERSONS)}

# ---------- Reload watchlist ----------
@app.post("/reload")
def reload_watchlist_endpoint():
    """Reload the watchlist from NPZ cache without restarting (FAST)"""
    import time
    start = time.time()
    
    try:
        print("\n" + "="*60)
        print("[ENDPOINT] /reload called")
        print("="*60)
        
        count = reload_watchlist()
        
        elapsed = time.time() - start
        print(f"[ENDPOINT] Reload completed in {elapsed:. 2f}s")
        
        return {
            "status": "success",
            "message": "Watchlist reloaded from NPZ cache",
            "watchlist_size": count,
            "reload_time_seconds": round(elapsed, 2)
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"[ERROR] Reload failed after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": str(e),
            "reload_time_seconds": round(elapsed, 2)
        }

# ---------- Config endpoints ----------
@app.get("/config")
def get_config():
    return {
        "Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh,
        "w_hdic": w_hdic, "T_final": T_final,
        "admin_api": ADMIN_API
    }

@app.post("/config")
def update_config(payload: dict):
    global Tnh, Thdic, w_nh, w_hdic, T_final, ADMIN_API
    # Safely parse floats and update globals
    if "Tnh" in payload:
        try: Tnh = float(payload["Tnh"])
        except: pass
    if "Thdic" in payload:
        try: Thdic = float(payload["Thdic"])
        except: pass
    if "w_nh" in payload:
        try:
            v = float(payload["w_nh"])
            if 0.0 <= v <= 1.0: w_nh = v
        except: pass
    if "w_hdic" in payload:
        try:
            v = float(payload["w_hdic"])
            if 0.0 <= v <= 1.0: w_hdic = v
        except: pass
    if "T_final" in payload:
        try: T_final = float(payload["T_final"])
        except: pass
    if "admin_api" in payload:
        ADMIN_API = str(payload["admin_api"])

    cfg = {
        "Tnh": Tnh, "Thdic": Thdic, "w_nh": w_nh, "w_hdic": w_hdic,
        "T_final": T_final, "port": PORT,
        "log_path": str(LOGPATH.relative_to(Path(__file__).parent)),
        "admin_api": ADMIN_API
    }
    CFG_PATH.write_text(yaml.safe_dump(cfg))
    return {"ok": True, "config": cfg}

# ---------- Single-frame match ----------
@app.post("/match")
async def match(file: UploadFile = File(...)):
    img = np.frombuffer(await file.read(), dtype=np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if frame is None:
        return {"decision": "ERROR", "error": "Invalid image"}

    result = match_frame(
        frame_bgr=frame,
        Tnh=Tnh, Thdic=Thdic,
        w_nh=w_nh, w_hdic=w_hdic,
        fused_th=T_final
    )
    result["frames"] = 1
    log_match(result)

    # Send to admin on MATCH
    if result.get("decision") == "MATCH":
        pid = result.get("person_id") or "unknown"
        sfinal = (result.get("scores") or {}).get("Sfinal", 0.0)
        send_manual_alert(pid, sfinal, frame)

    return result

# ---------- Multi-frame match ----------
@app.post("/match_multi")
async def match_multi(files: List[UploadFile] = File(...)):
    frames_data = [(i, await uf.read(), uf.filename) for i, uf in enumerate(files)]

    def process_single(idx, img_bytes, filename):
        try:
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return None

            result = match_frame(
                frame_bgr=frame,
                Tnh=Tnh, Thdic=Thdic,
                w_nh=w_nh, w_hdic=w_hdic,
                fused_th=T_final,
            )

            scores = result.get("scores", {}) or {}
            Sfinal = scores.get("Sfinal")
            d_nh = scores.get("d_nh")
            d_hdic = scores.get("d_hdic")
            pid = result.get("person_id")

            ok = (
                d_nh is not None and d_hdic is not None and Sfinal is not None and
                d_nh < Tnh and d_hdic < Thdic and Sfinal >= T_final
            )
            # One short line per frame
            try:
                if d_nh is not None and d_hdic is not None and Sfinal is not None:
                    print(f"Frame {idx+1}: d_nh={int(d_nh)}, d_hdic={int(d_hdic)}, Sfinal={float(Sfinal):.3f} -> {'MATCH' if ok else 'NO_MATCH'}")
                else:
                    print(f"Frame {idx+1}: NO_FACE or ERROR")
            except:
                pass

            return {
                "index": idx + 1,
                "filename": filename,
                "ok": ok,                        # <- authoritative for majority
                "decision": "MATCH" if ok else "NO_MATCH",
                "Sfinal": float(Sfinal) if Sfinal is not None else None,
                "d_nh": int(d_nh) if d_nh is not None else None,
                "d_hdic": int(d_hdic) if d_hdic is not None else None,
                "person_id": pid,
            }

        except Exception:
            return None

    # Process frames in parallel
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(filter(None, ex.map(lambda args: process_single(*args), frames_data)))

    if not results:
        return {"decision": "NO_FACE", "frames": 0}

    total = len(results)
    match_count = sum(1 for r in results if r.get("ok"))
    needed = ceil(total / 2)  # majority, for 5 → 3
    majority_decision = "MATCH" if match_count >= needed else "NO_MATCH"

    best_frame = max(results, key=lambda x: x.get("Sfinal", 0) or 0.0)
    best_score = best_frame.get("Sfinal", 0.0)
    best_pid = best_frame.get("person_id", None)
    best_idx = best_frame["index"] - 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    final_result = {
        "method": f"majority-of-{total}",
        "frames": total,
        "match_frames": match_count,
        "match_ratio": round(match_count / total, 2),
        "decision": majority_decision,
        "best_score": round(best_score, 3),
        "best_person_id": best_pid if match_count >= needed else None,
        "frame_details": results,
        "timestamp": timestamp,
    }

    log_match(final_result)

    if majority_decision == "MATCH" and best_pid:
        try:
            _, best_bytes, _ = frames_data[best_idx]
            enqueue_case(best_pid, best_score, best_bytes)
            send_manual_alert(
                person_id=best_pid,
                score=best_score,
                image_bytes=best_bytes,
                timestamp=timestamp,
            )
        except Exception:
            pass

    return final_result


# ✅ NEW: Admin feedback check endpoint
@app.get("/check_status")
def check_admin_status(person_id: str, timestamp: str):
    try:
        r = requests.get(
            f"{ADMIN_API}/manual_check/status",
            params={"person_id": person_id, "timestamp": timestamp},
            timeout=5,
            proxies={"http": None, "https": None}
        )
        return r.json()
    except Exception as e:
        print("[WARN] Failed to query admin for status:", e)
        return {"status": "unknown", "error": str(e)}


# ---------- Entry ----------
if __name__ == "__main__":
    uvicorn.run(
        "software_builds.field_client.backend.main:app",
        host="127.0.0.1",
        port=PORT,
        reload=True,
    )
