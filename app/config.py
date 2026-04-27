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
    #   off    → CSV+DataCache only (current behavior, default)
    #   shadow → DataCache primary + LeagueDataLoader runs alongside, dumps
    #            DataFrames to disk for diff vs CSV-mode baseline (Phase 4)
    #   on     → LeagueDataLoader is the source of truth (Phase 6 cutover)
    USE_DB_LOADER = os.getenv("USE_DB_LOADER", "off").lower()