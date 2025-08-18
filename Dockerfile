# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first (build cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Cloud Run listens on $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

# Use gunicorn to serve Flask app "main:app"
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--timeout", "120", "main:app"]
