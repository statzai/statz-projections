"""
Shared resolvers for league-projection writers: competition_id lookup,
current season_id lookup, and the split-format gate. Used by
`predicted_table_repo` and `league_position_repo` (and any future
league-grain writer that needs the same season / format machinery).

Was `league_outcome_repo.py` until 2026-05-19, when the rule-driven
`league_projection_outcomes` write was retired (statz read side switched
to the per-position distribution in `league_position_probabilities` —
see docs/league-projections-redesign.md). Renamed to reflect that what
remains is just the season / format helpers.
"""
import logging
import re

from app.repository.db_utils import fetch_all

logger = logging.getLogger("league_season_helpers")

# Stage-name patterns identifying knockout / promotion-playoff stages —
# anything that is NOT regular league play. Stages whose name matches
# this regex are excluded from the "league stage" set used by
# is_league_stage_complete below. Calibrated against the 2026-05-19 prod
# stage audit:
#   Excluded ("Promotion Play-offs - Semi-finals" / "- Final",
#             "Conference League Play-offs - …",
#             "Apertura, Quarterfinals" / "Semifinals" / "Final",
#             "Apertura - Reclasificación", "Apertura, Play In",
#             "Clausura - Quarter-finals" / "Semi-finals" / "Final")
#   Kept as league ("Regular Season", "Apertura", "Clausura", "2nd Phase")
#
# The terminal `\bfinal\b\s*$` anchor only matches "Final" at the end of
# the stage name, so a hypothetical "Final Round"-style regular stage
# would NOT be excluded.
_NON_LEAGUE_STAGE_RE = re.compile(
    r'play.?off|play\s?in|quarter.?final|semi.?final|reclasif|knockout|\bfinal\b\s*$',
    re.IGNORECASE,
)

# Brazil Serie A is keyed by a fixed competition_id — mirrors the special
# case in predicted_table_repo.insert_predicted_table_async.
_BRAZIL_COMPETITION_ID = 648


def resolve_competition_id(comps, league):
    """League name -> competition_id, mirroring insert_predicted_table_async."""
    if league == 'Brazil Serie A':
        return _BRAZIL_COMPETITION_ID
    match = comps.loc[comps['name'] == league, 'id']
    if match.empty:
        raise Exception(f"League {league} not found in comps")
    return int(match.iloc[0])


async def resolve_current_season_id(competition_id):
    """The current season id for a competition, or None if none is flagged."""
    rows = await fetch_all(
        "SELECT id FROM seasons WHERE competition_id = %s AND is_current = 1 "
        "ORDER BY id DESC LIMIT 1",
        (competition_id,),
    )
    return int(rows[0][0]) if rows else None


async def is_split_format_competition(competition_id, season_id):
    """A competition is split-format if its current season carries league
    standings across more than one stage — the Scottish Premiership
    (top-6 / bottom-6) and the Austrian / Belgian / Danish / Greek
    "Championship Round" + "Relegation Round" formats.

    Counts distinct stage_id among STANDINGS rows only, so knockout
    promotion play-offs (no standings -> one stage) and CL / EL (many
    stages, standings only on the league stage) correctly read as one
    stage and are NOT skipped. See docs/league-projections-redesign.md.
    """
    rows = await fetch_all(
        "SELECT COUNT(DISTINCT stage_id) FROM standings "
        "WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    return bool(rows) and rows[0][0] is not None and int(rows[0][0]) > 1


async def is_league_stage_complete(competition_id, season_id):
    """True if the competition's regular league stage has no upcoming
    fixtures.

    A "league stage" is any stage whose name does NOT match the knockout
    regex above. Used as a gate so the league-projection writes stop
    refreshing once the regular-season fixtures are done — even if
    promotion play-off / cup-style knockout fixtures remain (e.g. the
    Championship's Promotion Play-offs Final after the regular season
    ends 2026-05-02).

    Returns False (i.e. "keep projecting") on any of:
      - season_id is None,
      - stages table has no rows for this season,
      - every stage looks like a knockout (no league stage at all).
    These cases defer to the other gates and the standard scheduler.
    """
    if season_id is None:
        return False

    stage_rows = await fetch_all(
        "SELECT id, name FROM stages WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    league_stage_ids = [
        int(sid) for sid, name in stage_rows
        if name and not _NON_LEAGUE_STAGE_RE.search(name)
    ]
    if not league_stage_ids:
        return False

    placeholders = ','.join(['%s'] * len(league_stage_ids))
    rows = await fetch_all(
        f"SELECT COUNT(*) FROM fixtures "
        f"WHERE competition_id = %s AND season_id = %s "
        f"AND stage_id IN ({placeholders}) AND kickoff_datetime > NOW()",
        (competition_id, season_id, *league_stage_ids),
    )
    return bool(rows) and rows[0][0] is not None and int(rows[0][0]) == 0
