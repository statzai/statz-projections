FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/projection-outputs

EXPOSE 8000

# Single worker: the _projection_running lock in routes.py is a Python
# module global and therefore PER-WORKER under gunicorn. With 2 workers
# the lock was a false promise — 2026-04-24 saw two concurrent fetches
# hit different workers, each held its own lock, both raced to write
# fixture_team_stats.csv and corrupted ~1M rows. Switched to 1 worker
# so the lock is genuinely process-wide. Read endpoints stay snappy
# (they're sub-second); long writes were already serialised anyway.
CMD ["gunicorn", "app.main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "1800"]
