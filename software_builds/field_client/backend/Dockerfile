# Field Client Backend (FastAPI) on port 5001
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# If OpenCV or similar is used, these libs are commonly required
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from repo root
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire repo so backend can import shared modules
# (common, preprocess, neuralhash, hdic, fusion, db, etc.)
COPY . /app

# Ensure Python can locate the modules from repo root
ENV PYTHONPATH=/app

# Move to the backend directory (expects main:app here)
WORKDIR /app/software_builds/field_client/backend

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]