FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/projection-outputs

EXPOSE 8000

# Two workers. Tried single-worker on 2026-04-24 to make the in-memory
# _projection_running lock truly process-wide, but the fetch-data OOM'd
# under a single worker with only 4.7GB free host RAM (7.8M row
# DataFrame needs 3-5GB). Cross-worker serialisation now handled by
# the file-lock in routes.py (/tmp/_projection.lock) — survives worker
# boundaries AND restarts without risking memory blowups.
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "1800"]
