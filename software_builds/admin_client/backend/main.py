# software_builds/admin_client/backend/main.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from admin_api import router as admin_router

app = FastAPI(title="Hybrid Admin Backend")

# Admin UI will run on 5174 in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(admin_router, prefix="")
