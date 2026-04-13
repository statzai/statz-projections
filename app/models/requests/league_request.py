from typing import List, Optional
from pydantic import BaseModel

class LeagueRequest(BaseModel):
    league: str
    fixture_ids: Optional[List[int]] = None
