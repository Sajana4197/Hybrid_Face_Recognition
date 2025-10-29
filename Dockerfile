FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# If using opencv-python-headless, try fewer system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip

# CPU-only PyTorch (avoid huge GPU wheels)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 --no-cache-dir

RUN pip install -r requirements.txt --no-cache-dir

COPY . /app

ENV HOST=0.0.0.0
ENV PORT=5001
EXPOSE 5001 5002

CMD ["sh", "-c", "uvicorn software_builds.field_client.backend.main:app --host ${HOST} --port ${PORT}"]