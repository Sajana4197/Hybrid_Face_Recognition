from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .matcher import match_bytes

app = FastAPI(title="Field API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/match")
async def match(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return match_bytes(data)
    except Exception as e:
        raise HTTPException(400, f"match failed: {e}")
