FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run listens on $PORT
ENV PORT=8080

# 1 worker, 4 threads is a good baseline on Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 900 model:app

