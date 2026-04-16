from typing import List, Optional
from pydantic import BaseModel

class LeagueRequest(BaseModel):
    league: str
    fixture_ids: Optional[List[int]] = None
    # "full" (default) runs the full projection pipeline including accuracy
    # dataset gap-fill + metrics calculation. "refresh" skips those analysis
    # blocks since the 1:35pm scheduled refresh run doesn't need to rebuild
    # accuracy numbers that were already computed by the morning 2am run.
    # See memory/projections_system.md for the full justification.
    mode: Optional[str] = "full"
    # When true, fetch fresh data from source DB before projecting. Used by
    # the admin panel "Run Now" button so users get current data without
    # having to manually trigger fetch first. Scheduled runs don't use this
    # (the scheduler handles fetch separately with polling + timeout).
    fetch_first: Optional[bool] = False
