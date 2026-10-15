FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Only the serving code goes in the image. The model is pulled from GCS at
# startup, so a new model never needs a new image.
COPY app.py .

RUN useradd --create-home appuser
USER appuser

# Cloud Run injects $PORT. One worker with threads keeps a single model copy in memory.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app