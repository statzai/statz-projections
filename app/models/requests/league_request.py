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
    # Lean tournament-only re-sim. When True (and no fixture_ids), the run
    # executes ONLY the bracket-wide steps — ratings (1), 1X2 (2), tournament
    # sim (3) and tournament-player/top-scorer totals (7) — and SKIPS the
    # expensive per-fixture stat/fantasy steps (4/5/6/6b). ~15-20s vs ~3.5min.
    # Used by the kickoff-timed post-match finalizer to refresh advancement %,
    # power ratings and the scorer race right after a result lands, without
    # re-deriving every player's per-fixture lines (those refresh on the
    # twice-daily full passes). Must be paired with an empty fixture list.
    lean: Optional[bool] = False
