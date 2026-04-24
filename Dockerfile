FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/projection-outputs

EXPOSE 8000

# Single worker. With the 2-season fetch, fixture_player_stats.csv has
# 15M rows (~4GB in pandas). 2 workers each holding their own DataCache
# doubled memory pressure and OOM'd on the 2nd worker's cache load.
# Single worker means: one DataCache instance, loaded once, reused for
# all requests. Streaming fetch (SSCursor) means the fetch itself only
# uses ~60MB, so we don't need a second worker for throughput.
# The file-lock in routes.py is now redundant for cross-worker races
# (only one worker exists) but stays as a safety net for concurrent
# requests on the single worker.
CMD ["gunicorn", "app.main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "1800"]
