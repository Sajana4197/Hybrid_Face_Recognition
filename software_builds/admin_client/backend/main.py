# software_builds/admin_client/backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Allow running as a package ("backend.main") or a script ("python main.py")
try:
    from .admin_api import router as admin_router, UPLOADS_DIR
except ImportError:
    from admin_api import router as admin_router, UPLOADS_DIR  # type: ignore

app = FastAPI(title="Admin API", version="1.0")

# CORS: wide-open for dev; lock down in prod if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mount for uploaded verification images
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# API routes
app.include_router(admin_router)

@app.get("/health")
def health():
    return {"ok": True}
