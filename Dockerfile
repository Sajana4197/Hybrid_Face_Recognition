FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only needed system packages. Remove ffmpeg if not actually used.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in two steps for better caching.
# 1) Install torch/torchvision CPU wheels explicitly.
#    IMPORTANT: Do NOT list torch/torchvision in requirements.txt to avoid duplicate resolution/downloads.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --no-compile \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 && \
    python -m pip install --no-cache-dir --no-compile -r /app/requirements.txt

# 2) Copy only the code needed at runtime so cache holds if unrelated files change.
COPY software_builds/ /app/software_builds/
COPY fusion/ /app/fusion/
COPY preprocess/ /app/preprocess/
COPY neuralhash/ /app/neuralhash/
COPY hdic/ /app/hdic/

ENV HOST=0.0.0.0 \
    PORT=5001

EXPOSE 5001 5002

# docker-compose sets the command per service; this default is safe if run directly.
CMD ["uvicorn", "software_builds.field_client.backend.main:app", "--host", "0.0.0.0", "--port", "5001"]