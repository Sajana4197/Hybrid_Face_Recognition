from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from admin_api import router as admin_router

# ============================================================
# Main FastAPI App
# ============================================================
app = FastAPI(title="Admin Manual Verification API")

# CORS setup for UI (React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://192.168.1.2:5173",  # Replace with Laptop B's actual IP
        "http://192.168.1.2:5174",  # Replace with Laptop B's actual IP
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Include routers
# ============================================================
app.include_router(admin_router, prefix="")

# ============================================================
# ✅ Serve uploaded images statically
# ============================================================
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ============================================================
# Health check
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)