import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT"))
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")

    # Source Database
    SOURCE_DB_HOST = os.getenv("SOURCE_DB_HOST")
    SOURCE_DB_PORT = int(os.getenv("SOURCE_DB_PORT"))
    SOURCE_DB_USER = os.getenv("SOURCE_DB_USER")
    SOURCE_DB_PASSWORD = os.getenv("SOURCE_DB_PASSWORD")
    SOURCE_DB_NAME = os.getenv("SOURCE_DB_NAME")


    # Local Database
    LOCAL_DB_HOST = os.getenv("LOCAL_DB_HOST")
    LOCAL_DB_PORT = int(os.getenv("LOCAL_DB_PORT"))
    LOCAL_DB_USER = os.getenv("LOCAL_DB_USER")
    LOCAL_DB_PASSWORD = os.getenv("LOCAL_DB_PASSWORD")
    LOCAL_DB_NAME = os.getenv("LOCAL_DB_NAME")

    MIN_POOL_SIZE = int(os.getenv("MIN_POOL_SIZE", 1))
    MAX_POOL_SIZE = int(os.getenv("MAX_POOL_SIZE", 10))

    # Direct DB Query Migration feature flag.
    #   on     → LeagueDataLoader is the source of truth (DEFAULT, Phase 6 cutover 2026-04-28)
    #   off    → CSV+DataCache only (legacy fallback, requires fresh CSVs via fetch-data)
    #   shadow → DataCache primary + LeagueDataLoader alongside, snapshots to /tmp for diffing
    # Cutover proven on 24-league Run All Leagues at 01:03–01:49 BST 2026-04-28
    # (46 min total, zero failures, no OOMs) following an earlier ~3.5hr migration
    # session that exercised every code path. To roll back temporarily set
    # USE_DB_LOADER=off in .env and restart, but note: the 4GB DataCache will
    # OOM the worker under any host memory pressure (the original problem the
    # migration solved).
    USE_DB_LOADER = os.getenv("USE_DB_LOADER", "on").lower()