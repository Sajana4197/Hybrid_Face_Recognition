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
PERSONS = load_watchlists(REPO_ROOT)

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
    for k in ["Tnh","Thdic","w_nh","w_hdic","T_final","admin_api"]:
        if k in payload:
            val = payload[k]
            if k in ("w_nh","w_hdic") and not (0.0 <= float(val) <= 1.0):
                continue
            if k == "admin_api":
                ADMIN_API = str(val)
            else:
                locals()[k] = float(val)
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
                print(f"[WARN] Could not decode frame {filename}")
                return None

            result = match_frame(
                frame_bgr=frame,
                Tnh=Tnh, Thdic=Thdic,
                w_nh=w_nh, w_hdic=w_hdic,
                fused_th=T_final,
            )

            Sfinal = result.get("scores", {}).get("Sfinal", 0)
            d_nh = result.get("scores", {}).get("d_nh", None)
            d_hdic = result.get("scores", {}).get("d_hdic", None)
            decision = result.get("decision", "UNKNOWN")
            pid = result.get("person_id", None)

            print(f"[INFO] Frame {idx+1}: Decision={decision}, Sfinal={Sfinal:.3f}")

            return {
                "index": idx + 1,
                "filename": filename,
                "decision": decision,
                "Sfinal": float(Sfinal) if Sfinal is not None else None,
                "d_nh": d_nh,
                "d_hdic": d_hdic,
                "person_id": pid,
            }

        except Exception as e:
            print(f"[WARN] Failed to process frame {idx+1}: {e}")
            return None

    # Run all frames in parallel
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(filter(None, ex.map(lambda args: process_single(*args), frames_data)))

    if not results:
        return {"decision": "NO_FACE", "frames": 0}

    total = len(results)
    match_count = sum(1 for r in results if r["decision"] == "MATCH")
    majority_decision = "MATCH" if match_count >= 3 else "NO_MATCH"

    best_frame = max(results, key=lambda x: x.get("Sfinal", 0) or 0.0)
    best_score = best_frame.get("Sfinal", 0.0)
    best_pid = best_frame.get("person_id", None)
    best_idx = best_frame["index"] - 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    final_result = {
        "method": "majority-of-N",
        "frames": total,
        "match_frames": match_count,
        "match_ratio": round(match_count / total, 2),
        "decision": majority_decision,
        "best_score": round(best_score, 3),
        "best_person_id": best_pid,
        "frame_details": results,
        "timestamp": timestamp,
    }

    log_match(final_result)

    # ---- Handle a MATCH (send to Admin + store locally) ----
    if majority_decision == "MATCH" and best_pid:
        try:
            _, best_bytes, _ = frames_data[best_idx]
            enqueue_case(best_pid, best_score, best_bytes)  # Store locally

            # Send to Admin for manual confirmation
            send_manual_alert(
                person_id=best_pid,
                score=best_score,
                image_bytes=best_bytes,
                timestamp=timestamp,
            )
        except Exception as e:
            print("[WARN] Failed to prepare best frame for alert or enqueue:", e)

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
