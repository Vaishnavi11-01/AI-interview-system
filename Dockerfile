FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .
RUN mkdir -p /var/data/logs /var/data/output /var/data/uploads \
    && curl --fail --location --retry 3 \
      https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt \
      --output models/yolov8n-face.pt

EXPOSE 10000
CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 0"]